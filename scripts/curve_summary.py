"""Phase 3 acceptance script: bootstrap the live curve and run the RV screens.

    uv run python scripts/curve_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd  # noqa: E402

from cross_asset_engine.data.market_data import MarketData  # noqa: E402
from cross_asset_engine.settings import load_params  # noqa: E402
from cross_asset_engine.signals.base import pivot_to_panel  # noqa: E402
from cross_asset_engine.tenors import TENOR_YEARS, TENOR_ORDER  # noqa: E402
from cross_asset_engine.curve.bootstrap import bootstrap_zero_curve, implied_forward_curve  # noqa: E402
from cross_asset_engine.curve.rv_screens import (  # noqa: E402
    slope_screen, butterfly_screen, carry_rolldown_screen, latest_yields_from_panel,
)


def main() -> None:
    md = MarketData(project_root=PROJECT_ROOT)
    params = load_params()
    cparams = params["curve"]

    panel = pivot_to_panel(md.get_curve())
    latest = latest_yields_from_panel(panel)
    tenors = [t for t in TENOR_ORDER if t in latest]
    years = [TENOR_YEARS[t] for t in tenors]
    yields = [latest[t] for t in tenors]

    curve = bootstrap_zero_curve(years, yields, cparams["bootstrap_step_years"],
                                 cparams["interpolation"], max_maturity=max(years))
    fwd = implied_forward_curve(curve)

    print("=== BOOTSTRAPPED CURVE (par -> zero -> 6m fwd) ===")
    print(f"{'t(yr)':>6} {'par%':>7} {'zero%':>7} {'fwd%':>7} {'DF':>9}")
    for T in [0.5, 1, 2, 5, 10, 30]:
        i = min(range(len(curve.times)), key=lambda k: abs(curve.times[k] - T))
        print(f"{curve.times[i]:6.1f} {curve.par_yields_pct[i]:7.3f} {curve.zero_rates_pct[i]:7.3f} {fwd[i]:7.3f} {curve.discount_factors[i]:9.5f}")

    print("\n=== SLOPE SCREENS ===")
    for s in cparams["slopes"]:
        if s["short"] in panel and s["long"] in panel:
            r = slope_screen(panel, s["name"], s["short"], s["long"], cparams["zscore_window"])
            print(f"  {r.name}: {r.slope_bp:6.1f}bp  z={r.zscore:5.2f}  {r.signal}")
            print(f"     DV01-neutral: {r.hedge_ratio_short_per_long:.2f} units {r.short} per 1 unit {r.long}")

    print("\n=== BUTTERFLY SCREENS ===")
    for b in cparams["butterflies"]:
        r = butterfly_screen(panel, b["name"], b["wing_short"], b["belly"], b["wing_long"], cparams["zscore_window"])
        print(f"  {r.name}: {r.fly_bp:6.1f}bp  z={r.zscore:5.2f}  {r.signal}")

    print(f"\n=== CARRY + ROLLDOWN (horizon {cparams['carry_rolldown_horizon_months']}m, ranked per DV01) ===")
    horizon = cparams["carry_rolldown_horizon_months"] / 12.0
    points = carry_rolldown_screen(latest, horizon, cparams["interpolation"])
    print(f"{'tenor':>6} {'yield%':>7} {'roll_bp':>8} {'carry%':>7} {'roll%':>7} {'total%':>7} {'/DV01':>8}")
    for p in points:
        print(f"{p.tenor:>6} {p.yield_pct:7.3f} {p.roll_yield_bp:8.2f} {p.carry_return_pct:7.3f} "
              f"{p.rolldown_return_pct:7.3f} {p.total_return_pct:7.3f} {p.total_per_dv01:8.2f}")
    if points:
        print(f"\n  richest carry+roll per DV01: {points[0].tenor}   cheapest: {points[-1].tenor}")


if __name__ == "__main__":
    main()
