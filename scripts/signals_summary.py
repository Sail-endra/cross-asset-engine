"""Phase 2 acceptance script: run every signal on live data and print the
latest reads, so momentum / carry / regime output can be eyeballed for
sanity. Cache-first, so it reuses today's Phase 1 pulls.

    uv run python scripts/signals_summary.py
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
from cross_asset_engine.signals import SIGNAL_REGISTRY  # noqa: E402
from cross_asset_engine.signals import regime as regime_mod  # noqa: E402


def main() -> None:
    md = MarketData(project_root=PROJECT_ROOT)
    params = load_params()

    print("=== REGIME ===")
    for method in ("threshold", "markov"):
        r = regime_mod.current_regime(md, params, method=method)
        extra = f"vol={r.get('vol', float('nan')):.3f} z={r.get('z', float('nan')):.2f}" if method == "threshold" else f"p_high={r.get('p_high', float('nan')):.2f}"
        print(f"  {method:9s} -> {r['regime']:7s} ({extra}) as of {r['date'].date()}")

    for name in ("momentum", "carry"):
        print(f"\n=== {name.upper()} ===")
        results = SIGNAL_REGISTRY[name](md, params)
        rows = []
        for s in results:
            rows.append(
                {
                    "asset_class": s.asset_class,
                    "asset_id": s.asset_id,
                    "variant": s.variant,
                    "score": None if s.score is None or pd.isna(s.score) else round(s.score, 4),
                    "dir": s.direction,
                }
            )
        df = pd.DataFrame(rows)
        # show one representative variant per family to keep it readable
        if name == "momentum":
            df = df[df["variant"].isin(["ts_126d", "xs_126d"])]
        with pd.option_context("display.width", 140, "display.max_rows", 200):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
