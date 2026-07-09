"""Time-series and cross-sectional momentum.

Finance rationale
-----------------
Momentum is the most robust of the systematic style premia: assets that have
risen over the past 3-12 months tend to keep rising over the next month, and
vice versa. The classic time-series (trend) form goes long anything with a
positive trailing return and short anything negative; the cross-sectional
form ranks assets against each other and goes long the winners / short the
losers. Both are implemented here.

Sign convention for rates (the subtle part)
-------------------------------------------
Momentum must be measured on a *tradable total return*, not on a yield. A
Treasury yield falling is a bond price *rising* -- so naive momentum on the
yield level would get the sign exactly backwards. We therefore convert each
tenor's yield series into an approximate bond total-return index before
measuring momentum:

    daily_return_t  ~=  y_{t-1}/252            (carry: one day of yield accrual)
                        - ModDur_{t-1} * dy_t  (price P&L from the yield move)

so that a run of falling yields produces a rising return index and a positive
(long-duration) momentum signal. The modified duration is computed
analytically from the par-bond approximation; Phase 3 replaces it with the
exact DV01/duration from the bootstrapped curve, but the *sign* -- the thing
that matters for the signal -- is already correct here.

Equities and FX need no such treatment: an index level and an FX rate are
already prices, so momentum runs on them directly. (Equity indices are price
indices, excluding dividends -- a small positive drift that essentially never
flips a momentum sign; noted for completeness.)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    SignalResult,
    cross_sectional_zscore,
    pivot_to_panel,
    register_signal,
    trailing_return,
)
from ..tenors import TENOR_YEARS


def par_bond_modified_duration(yield_pct: float, tenor_years: float) -> float:
    """Modified duration of a par bond at `yield_pct` (percent) and maturity.

    For sub-year money-market tenors, treats the instrument as a zero-coupon
    bill: ModDur = t / (1 + y*t). For >= 1y, prices the annual-coupon par bond
    (coupon = yield, price = 100) from its cash flows and returns Macaulay
    duration / (1 + y). Used only to scale the yield move into a price return;
    the sign of the signal does not depend on getting duration exactly right.
    """
    y = yield_pct / 100.0
    if not np.isfinite(y) or y <= -0.99:
        return np.nan
    if tenor_years < 1.0:
        return tenor_years / (1.0 + y * tenor_years)
    n = int(round(tenor_years))
    times = np.arange(1, n + 1)
    cashflows = np.full(n, y * 100.0)
    cashflows[-1] += 100.0
    discount = (1.0 + y) ** (-times)
    pv = cashflows * discount
    price = pv.sum()  # ~100 for a par bond
    macaulay = (times * pv).sum() / price
    return macaulay / (1.0 + y)


def yields_to_total_return_index(yields_panel: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide panel of tenor yields (percent) into TR-index levels.

    Applies the carry-minus-duration*dy approximation per tenor and compounds
    the daily returns into an index starting at 100. Strictly backward-looking:
    the return on day t uses the yield and duration as of day t-1 and the yield
    change into t.
    """
    tr = pd.DataFrame(index=yields_panel.index, columns=yields_panel.columns, dtype=float)
    for tenor in yields_panel.columns:
        years = TENOR_YEARS.get(tenor)
        if years is None:
            continue
        y = yields_panel[tenor]
        dy = y.diff()
        # duration uses prior-day yield so nothing on day t peeks at day t
        moddur = y.shift(1).apply(lambda v: par_bond_modified_duration(v, years))
        carry = y.shift(1) / 100.0 / 252.0
        price_pnl = -moddur * (dy / 100.0)
        daily_ret = carry + price_pnl
        tr[tenor] = 100.0 * (1.0 + daily_ret.fillna(0.0)).cumprod()
    return tr


def _direction(x: float) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(np.sign(x))


def time_series_signals(
    prices: pd.DataFrame, asset_class: str, lookbacks: List[int], as_of: pd.Timestamp | None = None
) -> List[SignalResult]:
    """Time-series momentum: sign of each asset's own trailing return."""
    results: List[SignalResult] = []
    as_of = as_of or prices.index.max()
    for lb in lookbacks:
        ret = trailing_return(prices, lb)
        if as_of not in ret.index:
            continue
        row = ret.loc[as_of]
        for asset_id, score in row.items():
            results.append(
                SignalResult(
                    signal="momentum",
                    variant=f"ts_{lb}d",
                    asset_id=asset_id,
                    asset_class=asset_class,
                    date=as_of,
                    score=float(score) if pd.notna(score) else np.nan,
                    direction=_direction(score),
                    inputs={"lookback_days": lb, "trailing_return": _f(score)},
                )
            )
    return results


def cross_sectional_signals(
    prices: pd.DataFrame, asset_class: str, lookbacks: List[int], as_of: pd.Timestamp | None = None
) -> List[SignalResult]:
    """Cross-sectional momentum: z-score trailing returns across the group."""
    results: List[SignalResult] = []
    as_of = as_of or prices.index.max()
    for lb in lookbacks:
        ret = trailing_return(prices, lb)
        if as_of not in ret.index:
            continue
        z = cross_sectional_zscore(ret.loc[as_of])
        for asset_id, score in z.items():
            results.append(
                SignalResult(
                    signal="momentum",
                    variant=f"xs_{lb}d",
                    asset_id=asset_id,
                    asset_class=asset_class,
                    date=as_of,
                    score=float(score) if pd.notna(score) else np.nan,
                    direction=_direction(score),
                    inputs={
                        "lookback_days": lb,
                        "trailing_return": _f(ret.loc[as_of, asset_id]),
                        "xs_zscore": _f(score),
                    },
                )
            )
    return results


def _f(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


@register_signal("momentum")
def compute_momentum(market_data, params: Dict, as_of: pd.Timestamp | None = None) -> List[SignalResult]:
    """Momentum across every asset class, using the right price proxy for each.

    equities -> index levels; fx -> fx rates; rates -> the bond TR index built
    from yields (see module docstring). Returns both time-series and
    cross-sectional variants for all configured lookbacks.
    """
    lookbacks = params["signals"]["momentum"]["lookbacks"]
    out: List[SignalResult] = []

    eq = pivot_to_panel(market_data.get_equities())
    out += time_series_signals(eq, "equities", lookbacks, as_of)
    out += cross_sectional_signals(eq, "equities", lookbacks, as_of)

    fx = pivot_to_panel(market_data.get_fx())
    out += time_series_signals(fx, "fx", lookbacks, as_of)
    out += cross_sectional_signals(fx, "fx", lookbacks, as_of)

    yields = pivot_to_panel(market_data.get_curve())
    rates_tr = yields_to_total_return_index(yields)
    out += time_series_signals(rates_tr, "rates", lookbacks, as_of)
    out += cross_sectional_signals(rates_tr, "rates", lookbacks, as_of)

    return out
