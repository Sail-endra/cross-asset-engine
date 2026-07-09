"""Carry: the return earned if nothing moves.

Finance rationale
-----------------
Carry is the second great systematic premium alongside momentum. Its exact
definition differs by asset class, and getting each one right is the point:

- FX carry     = short-rate differential. Long a high-yield currency funded
                 in a low-yield one earns the rate gap (r_base - r_quote) as
                 long as the spot rate doesn't move against you.
- Rates carry  = yield + rolldown. Holding a bond earns its yield (carry); on
                 an upward-sloping curve it also gains as it "rolls down" to
                 shorter, lower-yielding maturities. The rolldown piece needs
                 the bootstrapped curve, so it is a clean hook filled in
                 Phase 3; here we expose the yield-carry component.
- Equity carry = dividend yield - financing rate. You collect dividends and
                 pay to finance the position. The dividend-yield input has no
                 free daily source in this project's data spine, so equity
                 carry is a documented hook (mechanism implemented, data
                 pending) rather than fabricated.
- Credit carry = the OAS itself. The spread is literally the compensation for
                 bearing default/liquidity risk -- the carry of the position.

Interaction with regime (see regime.py)
---------------------------------------
Carry and momentum struggle at different times, and carry specifically bleeds
in risk-off regimes when volatility spikes. `apply_regime_scaling` multiplies
carry exposure by the regime's carry_scale so the book de-risks into stress.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import SignalResult, pivot_to_panel, register_signal
from . import regime as regime_mod


def _direction(x: float) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(np.sign(x))


def credit_carry(market_data, as_of: Optional[pd.Timestamp] = None) -> List[SignalResult]:
    """Credit carry = OAS level (percent). Positive spread = positive carry."""
    credit = market_data.get_credit()
    panel = pivot_to_panel(credit)
    as_of = as_of or panel.index.max()
    if as_of not in panel.index:
        return []
    row = panel.loc[as_of]
    return [
        SignalResult(
            signal="carry",
            variant="credit_oas",
            asset_id=asset_id,
            asset_class="credit",
            date=as_of,
            score=float(oas) if pd.notna(oas) else np.nan,
            direction=_direction(oas),
            inputs={"oas_pct": _f(oas)},
        )
        for asset_id, oas in row.items()
    ]


def fx_carry(market_data, as_of: Optional[pd.Timestamp] = None) -> List[SignalResult]:
    """FX carry = r_base - r_quote, per configured pair.

    Short rates are forward-filled onto the FX trading calendar -- an explicit
    alignment of step-like policy rates, done here rather than in the loader.
    """
    fx_panel = pivot_to_panel(market_data.get_fx())
    as_of = as_of or fx_panel.index.max()

    short = market_data.get_short_rates()
    rate_panel = pivot_to_panel(short)
    # align policy rates to FX dates, forward-fill the step function
    rate_panel = rate_panel.reindex(fx_panel.index).ffill()

    results: List[SignalResult] = []
    for item in market_data.instruments["fx"]:
        base, quote = item["from_symbol"], item["to_symbol"]
        pair = item["asset_id"]
        if as_of not in rate_panel.index:
            continue
        r_base = rate_panel.at[as_of, base] if base in rate_panel.columns else np.nan
        r_quote = rate_panel.at[as_of, quote] if quote in rate_panel.columns else np.nan
        carry = r_base - r_quote
        results.append(
            SignalResult(
                signal="carry",
                variant="fx_rate_diff",
                asset_id=pair,
                asset_class="fx",
                date=as_of,
                score=float(carry) if pd.notna(carry) else np.nan,
                direction=_direction(carry),
                inputs={"r_base_pct": _f(r_base), "r_quote_pct": _f(r_quote), "base": base, "quote": quote},
            )
        )
    return results


def rates_carry(
    market_data, as_of: Optional[pd.Timestamp] = None, rolldown: Optional[Dict[str, float]] = None
) -> List[SignalResult]:
    """Rates carry = yield + rolldown. Rolldown is a Phase-3 hook (default 0).

    `rolldown` maps tenor -> rolldown in percent; when Phase 3's curve module
    is wired in it supplies this, and the score becomes true carry+rolldown.
    """
    rolldown = rolldown or {}
    panel = pivot_to_panel(market_data.get_curve())
    as_of = as_of or panel.index.max()
    if as_of not in panel.index:
        return []
    row = panel.loc[as_of]
    results = []
    for tenor, yld in row.items():
        rd = rolldown.get(tenor, 0.0)
        total = (yld + rd) if pd.notna(yld) else np.nan
        results.append(
            SignalResult(
                signal="carry",
                variant="rates_yield_rolldown",
                asset_id=tenor,
                asset_class="rates",
                date=as_of,
                score=float(total) if pd.notna(total) else np.nan,
                direction=_direction(total),
                inputs={"yield_pct": _f(yld), "rolldown_pct": rd, "rolldown_wired": bool(rolldown)},
            )
        )
    return results


def equity_carry(
    market_data,
    dividend_yields: Optional[Dict[str, float]] = None,
    financing_rate: Optional[float] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> List[SignalResult]:
    """Equity carry = dividend yield - financing rate.

    Mechanism implemented; `dividend_yields` (asset_id -> percent) has no free
    daily source in this project, so when it is not supplied this returns an
    empty list (a documented hook), never a made-up number. `financing_rate`
    defaults to the USD short rate if available.
    """
    if not dividend_yields:
        return []
    if financing_rate is None:
        short = market_data.get_short_rates()
        usd = short[short["asset_id"] == "USD"].sort_values("date")
        financing_rate = float(usd["value"].iloc[-1]) if len(usd) else 0.0
    as_of = as_of or pivot_to_panel(market_data.get_equities()).index.max()
    results = []
    for asset_id, dy in dividend_yields.items():
        carry = dy - financing_rate
        results.append(
            SignalResult(
                signal="carry",
                variant="equity_div_minus_financing",
                asset_id=asset_id,
                asset_class="equities",
                date=as_of,
                score=float(carry),
                direction=_direction(carry),
                inputs={"dividend_yield_pct": dy, "financing_rate_pct": financing_rate},
            )
        )
    return results


def apply_regime_scaling(
    carry_results: List[SignalResult],
    market_data,
    params: Dict,
    method: str = "threshold",
    as_of: Optional[pd.Timestamp] = None,
) -> List[SignalResult]:
    """Scale carry exposure by the current regime's carry_scale multiplier.

    Records both the raw and scaled score in `inputs` so the gating is fully
    auditable in the briefing. Direction (sign) is unchanged -- only size.
    """
    reg = regime_mod.current_regime(market_data, params, method=method, as_of=as_of)
    scale = regime_mod.carry_scale(reg["regime"], params)
    for r in carry_results:
        raw = r.score
        r.inputs["regime"] = reg["regime"]
        r.inputs["regime_scale"] = scale
        r.inputs["raw_score"] = _f(raw)
        r.inputs["scaled_score"] = None if raw is None or pd.isna(raw) else float(raw * scale)
    return carry_results


def _f(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _rates_rolldown(market_data, params: Dict) -> Dict[str, float]:
    """Rolldown per tenor from the curve module (Phase 3), for the carry hook.

    Imported lazily so the signals package has no hard dependency on the curve
    package at import time. Returns {} if the curve can't be built.
    """
    try:
        from ..curve.rv_screens import latest_yields_from_panel, rolldown_for_carry

        panel = pivot_to_panel(market_data.get_curve())
        latest = latest_yields_from_panel(panel)
        horizon_m = params.get("curve", {}).get("carry_rolldown_horizon_months", 3)
        interp = params.get("curve", {}).get("interpolation", "linear")
        return rolldown_for_carry(latest, horizon_m / 12.0, interp)
    except Exception:
        return {}


@register_signal("carry")
def compute_carry(market_data, params: Dict, as_of: pd.Timestamp | None = None) -> List[SignalResult]:
    """All carry components, with regime scaling applied to the combined set.

    Rates carry now includes rolldown sourced from the Phase 3 curve module,
    closing the hook left open in Phase 2.
    """
    results: List[SignalResult] = []
    results += credit_carry(market_data, as_of)
    results += fx_carry(market_data, as_of)
    results += rates_carry(market_data, as_of, rolldown=_rates_rolldown(market_data, params))
    results += equity_carry(market_data, as_of=as_of)  # hook: empty unless div yields supplied
    apply_regime_scaling(results, market_data, params, as_of=as_of)
    return results
