"""Phase 1 acceptance script: pull every configured series and print a
summary table of latest levels and day-over-day changes.

Run from the repo root:

    uv run python scripts/pull_data_summary.py

Requires FRED_API_KEY and ALPHAVANTAGE_API_KEY in a local .env file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from cross_asset_engine.data.exceptions import DataFetchError  # noqa: E402
from cross_asset_engine.data.market_data import MarketData  # noqa: E402


def main() -> None:
    md = MarketData(project_root=PROJECT_ROOT)
    try:
        summary = md.latest_levels_and_changes()
    except DataFetchError as e:
        print(f"DATA PULL FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    with pd_option_context():
        print(summary.to_string(index=False))

    print(f"\n{len(summary)} series pulled as of {md.as_of.isoformat()}.")
    for asset_class, group in summary.groupby("asset_class"):
        print(f"  {asset_class}: {len(group)} series")


def pd_option_context():
    import pandas as pd

    return pd.option_context("display.width", 120, "display.max_rows", 100)


if __name__ == "__main__":
    main()
