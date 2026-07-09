"""Alpha Vantage loader for equity/ETF and FX daily series.

Endpoints and response shapes verified live against alphavantage.co on
2026-07-08: TIME_SERIES_DAILY for equities, FX_DAILY for FX pairs. The free
tier is capped at 25 requests/day total (this changed from the historical
500/day figure -- re-verify against alphavantage.co/premium if this project
is revisited), so every call here is cache-first: a symbol already pulled
today is never re-requested.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Optional

import pandas as pd
import requests

from .cache import RawCache
from .exceptions import DataFetchError
from .schema import make_tidy_frame

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# The free tier asks for no more than ~1 request/second. We space real
# network calls a little beyond that to stay clear of the burst limit.
# Cache hits are never throttled, so same-day reruns stay instant.
_MIN_SECONDS_BETWEEN_REQUESTS = 1.2


class AlphaVantageLoader:
    def __init__(
        self,
        cache: RawCache,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise DataFetchError("ALPHAVANTAGE_API_KEY is not set. Add your key to .env")
        self.cache = cache
        self.session = session or requests.Session()
        self._last_request_ts: float | None = None

    def _throttle(self) -> None:
        """Sleep just enough to keep real network calls under the burst limit."""
        if self._last_request_ts is not None:
            elapsed = time.monotonic() - self._last_request_ts
            wait = _MIN_SECONDS_BETWEEN_REQUESTS - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def fetch_equity_daily(
        self, symbol: str, asset_id: str, as_of: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch daily close prices for an equity/ETF symbol (fallback path).

        NOTE: not used by MarketData -- US equities are sourced from FRED for
        long history (see instruments.yaml). This method is kept as a working
        fallback for any short-history / snapshot equity need. It requests
        outputsize=compact (~100 trading days) because outputsize=full is a
        premium-only feature on the free tier as of 2026.
        """
        payload = self._get_or_fetch(
            kind="equity",
            key=symbol,
            as_of=as_of or date.today(),
            params={"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "compact"},
        )
        series = payload.get("Time Series (Daily)")
        if not series:
            raise DataFetchError(
                f"Alpha Vantage returned no daily series for symbol={symbol!r}: {payload}"
            )
        dates = list(series.keys())
        closes = [row["4. close"] for row in series.values()]
        return make_tidy_frame(dates, asset_id, closes, source="alpha_vantage")

    def fetch_fx_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        asset_id: str,
        as_of: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch daily close rates for an FX pair (from_symbol/to_symbol)."""
        pair_key = f"{from_symbol}{to_symbol}"
        payload = self._get_or_fetch(
            kind="fx",
            key=pair_key,
            as_of=as_of or date.today(),
            params={
                "function": "FX_DAILY",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "outputsize": "full",
            },
        )
        series = payload.get("Time Series FX (Daily)")
        if not series:
            raise DataFetchError(
                f"Alpha Vantage returned no FX series for {from_symbol}/{to_symbol}: {payload}"
            )
        dates = list(series.keys())
        closes = [row["4. close"] for row in series.values()]
        return make_tidy_frame(dates, asset_id, closes, source="alpha_vantage")

    def _get_or_fetch(self, kind: str, key: str, as_of: date, params: dict) -> dict:
        cache_key = f"{kind}_{key}"
        payload = self.cache.get("alpha_vantage", cache_key, as_of)
        if payload is not None:
            return payload

        full_params = {**params, "apikey": self.api_key, "datatype": "json"}
        self._throttle()
        resp = self.session.get(ALPHA_VANTAGE_BASE_URL, params=full_params, timeout=20)
        try:
            payload = resp.json()
        except ValueError as e:
            raise DataFetchError(
                f"Alpha Vantage response for key={key!r} was not valid JSON: {resp.text[:200]}"
            ) from e

        if "Error Message" in payload:
            raise DataFetchError(f"Alpha Vantage error for key={key!r}: {payload['Error Message']}")
        if "Note" in payload or "Information" in payload:
            # Rate-limit / invalid-key responses come back as HTTP 200 with one
            # of these keys instead of data -- must not be treated as empty-but-ok.
            msg = payload.get("Note") or payload.get("Information")
            raise DataFetchError(f"Alpha Vantage rate-limit/info response for key={key!r}: {msg}")

        self.cache.put("alpha_vantage", cache_key, as_of, payload)
        return payload
