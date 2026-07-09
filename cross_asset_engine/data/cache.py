"""Dated disk cache for raw vendor API responses.

Two reasons this exists, both load-bearing for the project's constraints:

1. Reproducibility: caching by calendar date means a backtest or briefing
   re-run on the same day reads the exact same raw payload it read the first
   time, rather than a fresh network response that might differ.
2. Rate limits: Alpha Vantage's free tier is capped at 25 requests/day. A
   loader that checks the cache before every network call means re-running
   the pipeline (e.g. while debugging a signal) never burns quota twice for
   the same symbol on the same day.

The cache stores raw, unparsed vendor JSON -- not the tidied DataFrame -- so
that a parsing bug can be fixed and the cache replayed without re-fetching.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional


class RawCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, key: str, as_of: date) -> Path:
        safe_key = key.replace("/", "-").replace(" ", "_")
        dir_path = self.root / source / safe_key
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{as_of.isoformat()}.json"

    def get(self, source: str, key: str, as_of: date) -> Optional[Any]:
        """Return the cached payload for (source, key, as_of), or None if absent."""
        path = self._path(source, key, as_of)
        if not path.exists():
            return None
        with path.open("r") as f:
            return json.load(f)

    def put(self, source: str, key: str, as_of: date, payload: Any) -> None:
        """Persist a raw payload for (source, key, as_of)."""
        path = self._path(source, key, as_of)
        with path.open("w") as f:
            json.dump(payload, f)
