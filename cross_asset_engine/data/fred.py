"""FRED (Federal Reserve Economic Data) loader.

Source for the Treasury par-yield curve (DGS* series) and the ICE BofA
credit-spread series (BAMLC0A0CM, BAMLH0A0HYM2). Endpoint and response shape
verified live against api.stlouisfed.org on 2026-07-08 -- see README for how
to re-verify if this project is picked back up later and FRED has changed
something.

FRED marks non-trading days (weekends, holidays) with a literal "." value
rather than omitting the row. Those are dropped here as expected non-events,
not treated as missing data -- the real failure mode this project guards
against is an entire series coming back empty, or a gap between real
observations too large to be a holiday (see schema.check_no_large_gaps).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import pandas as pd
import requests

from .cache import RawCache
from .exceptions import DataFetchError
from .schema import make_tidy_frame

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredLoader:
    def __init__(
        self,
        cache: RawCache,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise DataFetchError(
                "FRED_API_KEY is not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env"
            )
        self.cache = cache
        self.session = session or requests.Session()

    def fetch_series(
        self,
        series_id: str,
        asset_id: str,
        observation_start: str = "1990-01-01",
        as_of: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch one FRED series and return it in the shared tidy schema."""
        as_of = as_of or date.today()
        cache_key = f"{series_id}_{observation_start}"
        payload = self.cache.get("fred", cache_key, as_of)
        if payload is None:
            payload = self._request(series_id, observation_start)
            self.cache.put("fred", cache_key, as_of, payload)

        if "error_code" in payload:
            raise DataFetchError(
                f"FRED error for series_id={series_id!r}: {payload.get('error_message')}"
            )

        observations = payload.get("observations")
        if not observations:
            raise DataFetchError(f"FRED returned no observations for series_id={series_id!r}")

        dates, values = [], []
        for obs in observations:
            if obs.get("value") == ".":
                continue  # non-trading day marker, not missing data
            dates.append(obs["date"])
            values.append(obs["value"])

        if not dates:
            raise DataFetchError(
                f"FRED series_id={series_id!r} returned only missing ('.') observations "
                f"in the requested window -- treat as a failed pull, not empty history"
            )

        return make_tidy_frame(dates, asset_id, values, source="fred")

    def _request(self, series_id: str, observation_start: str) -> dict:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }
        resp = self.session.get(FRED_BASE_URL, params=params, timeout=20)
        try:
            payload = resp.json()
        except ValueError as e:
            raise DataFetchError(
                f"FRED response for series_id={series_id!r} was not valid JSON: {resp.text[:200]}"
            ) from e
        if resp.status_code != 200 and "error_code" not in payload:
            raise DataFetchError(
                f"FRED HTTP {resp.status_code} for series_id={series_id!r}: {resp.text[:200]}"
            )
        return payload
