"""The daily pipeline entrypoint: pull data, compute everything, and write the
self-contained HTML briefing plus its dated JSON snapshot.

    uv run python scripts/run_daily.py            # rules-based narrative (default)
    uv run python scripts/run_daily.py --llm      # optional Anthropic narrative pass

This is the script GitHub Actions runs on a schedule (Phase 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from cross_asset_engine.briefing.builder import build_briefing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the daily cross-asset briefing.")
    parser.add_argument("--llm", action="store_true",
                        help="Use the Anthropic LLM narrative pass instead of rules-based templating.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: docs/).")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    paths = build_briefing(PROJECT_ROOT, use_llm=args.llm, output_dir=out)
    print(f"Briefing written: {paths['html']}")
    print(f"JSON snapshot:    {paths['json']}")


if __name__ == "__main__":
    main()
