"""Phase 4 acceptance script: backtest momentum, FX carry, and a curve trade
on live data; print gross/net metrics, the anti-overfitting lookback sweep,
and save an equity-curve chart + JSON snapshot.

    uv run python scripts/backtest_report.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd  # noqa: E402

from cross_asset_engine.data.market_data import MarketData  # noqa: E402
from cross_asset_engine.settings import load_params  # noqa: E402
from cross_asset_engine.backtest.engine import run_backtest  # noqa: E402
from cross_asset_engine.backtest.strategies import (  # noqa: E402
    build_default_backtests_inputs, momentum_strategy_equities,
)
from cross_asset_engine.backtest.report import (  # noqa: E402
    metrics_table, equity_curve_png_base64, results_to_snapshot, momentum_lookback_sweep,
)


def main() -> None:
    md = MarketData(project_root=PROJECT_ROOT)
    params = load_params()
    bt = params["backtest"]
    horizon, cost, rf = bt["horizon_days"], bt["cost_bps"], bt["risk_free_pct"]

    results = {}
    for name, (pos, px, units) in build_default_backtests_inputs(md, params).items():
        results[name] = run_backtest(name, pos, px, horizon, cost, rf, units)

    print(f"=== BACKTEST REPORT (horizon={horizon}d, cost={cost}bp, rf={rf}%) ===")
    with pd.option_context("display.width", 160):
        print(metrics_table(results).to_string(index=False))

    print("\n=== MOMENTUM LOOKBACK SWEEP (equities) -- report the spread, not the best cell ===")
    sweep = {}
    for lb in params["signals"]["momentum"]["lookbacks"]:
        pos, px = momentum_strategy_equities(md, lb)
        sweep[lb] = run_backtest(f"mom_{lb}", pos, px, horizon, cost, rf)
    print(momentum_lookback_sweep(sweep).to_string(index=False))

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    png_b64 = equity_curve_png_base64(results, net=True)
    (reports / "equity_curves.png").write_bytes(base64.b64decode(png_b64))
    snapshot = results_to_snapshot(results)
    (reports / "backtest_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    print(f"\nSaved chart -> reports/equity_curves.png")
    print(f"Saved snapshot -> reports/backtest_snapshot.json")


if __name__ == "__main__":
    main()
