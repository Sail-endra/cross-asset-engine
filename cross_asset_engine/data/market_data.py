"""Unified market-data accessor.

This is the one interface the rest of the system (signals, curve
construction, backtests, the briefing) is allowed to call. Nothing outside
this module should import FredLoader or AlphaVantageLoader directly, or know
a series ID or ticker -- that keeps a vendor swap (e.g. replacing Alpha
Vantage with a different equity source) a change to this file only.

Every get_* method returns the shared tidy schema (date, asset_id, value,
source) with an added asset_class column, and runs the gap check from
schema.py before returning.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ..settings import load_instruments, load_params
from .alphavantage import AlphaVantageLoader
from .cache import RawCache
from .fred import FredLoader
from .schema import check_no_large_gaps


class MarketData:
    def __init__(
        self,
        instruments_path: Optional[Path] = None,
        params_path: Optional[Path] = None,
        as_of: Optional[date] = None,
        project_root: Optional[Path] = None,
        fred_loader: Optional[FredLoader] = None,
        av_loader: Optional[AlphaVantageLoader] = None,
    ):
        self.instruments = load_instruments(instruments_path)
        self.params = load_params(params_path)
        self.as_of = as_of or date.today()

        root = project_root or Path.cwd()
        cache_dir = root / self.params["data"]["cache_dir"]
        cache = RawCache(cache_dir)

        # fred_loader/av_loader are injectable so tests can supply a fake
        # HTTP session without touching real network or environment keys.
        self._fred = fred_loader or FredLoader(cache)
        self._av = av_loader or AlphaVantageLoader(cache)
        self._max_gap_days = self.params["data"]["max_gap_calendar_days"]
        self._max_gap_days_monthly = self.params["data"].get(
            "max_gap_calendar_days_monthly", 45
        )
        self._fred_start = self.params["data"]["fred_observation_start"]

    def _validated(
        self, df: pd.DataFrame, asset_class: str, gap_days: Optional[int] = None
    ) -> pd.DataFrame:
        check_no_large_gaps(df, gap_days or self._max_gap_days)
        df = df.copy()
        df["asset_class"] = asset_class
        return df

    def get_curve(self) -> pd.DataFrame:
        """Full Treasury par-yield curve, one asset_id per tenor (e.g. '10Y')."""
        frames = [
            self._validated(
                self._fred.fetch_series(
                    point["series_id"], point["tenor"], self._fred_start, self.as_of
                ),
                asset_class="rates",
            )
            for point in self.instruments["rates"]["curve"]
        ]
        return pd.concat(frames, ignore_index=True)

    def get_short_rates(self) -> pd.DataFrame:
        """Per-currency short rates (percent) used by FX carry.

        asset_id is the currency code (USD, EUR, ...). Monthly series get a
        wider gap tolerance so the OECD foreign rates don't trip the daily
        gap check. Returned raw -- forward-filling to a trading calendar is
        the carry module's job, done explicitly where it aligns rates to FX.
        """
        frames = []
        for item in self.instruments["short_rates"]:
            is_monthly = item.get("frequency") == "monthly"
            gap = self._max_gap_days_monthly if is_monthly else self._max_gap_days
            frames.append(
                self._validated(
                    self._fred.fetch_series(
                        item["series_id"], item["currency"], self._fred_start, self.as_of
                    ),
                    asset_class="short_rate",
                    gap_days=gap,
                )
            )
        return pd.concat(frames, ignore_index=True)

    def get_credit(self) -> pd.DataFrame:
        """ICE BofA OAS series (investment grade and high yield), in basis points."""
        frames = [
            self._validated(
                self._fred.fetch_series(
                    item["series_id"], item["asset_id"], self._fred_start, self.as_of
                ),
                asset_class="credit",
            )
            for item in self.instruments["credit"]
        ]
        return pd.concat(frames, ignore_index=True)

    def get_equities(self) -> pd.DataFrame:
        """Daily index levels for the configured equity basket.

        Sourced from FRED rather than Alpha Vantage -- see instruments.yaml
        for why (free-tier equity history on Alpha Vantage is too short for
        backtests). Still returns the same tidy schema, so signals downstream
        neither know nor care that equities and rates share a vendor.
        """
        frames = [
            self._validated(
                self._fred.fetch_series(
                    item["series_id"], item["asset_id"], self._fred_start, self.as_of
                ),
                asset_class="equities",
            )
            for item in self.instruments["equities"]
        ]
        return pd.concat(frames, ignore_index=True)

    def get_fx(self) -> pd.DataFrame:
        """Daily close rates for the configured FX pair basket."""
        frames = [
            self._validated(
                self._av.fetch_fx_daily(
                    item["from_symbol"], item["to_symbol"], item["asset_id"], self.as_of
                ),
                asset_class="fx",
            )
            for item in self.instruments["fx"]
        ]
        return pd.concat(frames, ignore_index=True)

    def get_all(self) -> pd.DataFrame:
        """Every configured series, concatenated in the shared tidy schema."""
        return pd.concat(
            [self.get_curve(), self.get_credit(), self.get_equities(), self.get_fx()],
            ignore_index=True,
        )

    def latest_levels_and_changes(self) -> pd.DataFrame:
        """One row per asset_id: latest value, prior value, and day-over-day change.

        This is the acceptance-criteria summary table for the data spine --
        a quick sanity read on whether every series pulled cleanly and
        landed on a plausible level.
        """
        all_data = self.get_all()
        rows = []
        for asset_id, group in all_data.groupby("asset_id", sort=False):
            group = group.sort_values("date")
            if len(group) < 2:
                raise ValueError(
                    f"asset_id={asset_id!r} has fewer than 2 observations; "
                    "cannot compute a day-over-day change"
                )
            latest, prior = group.iloc[-1], group.iloc[-2]
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": latest["asset_class"],
                    "date": latest["date"],
                    "value": latest["value"],
                    "change": latest["value"] - prior["value"],
                    "source": latest["source"],
                }
            )
        summary = pd.DataFrame(rows)
        order = {"rates": 0, "credit": 1, "equities": 2, "fx": 3}
        summary["_order"] = summary["asset_class"].map(order)
        summary = summary.sort_values(["_order", "asset_id"]).drop(columns="_order")
        return summary.reset_index(drop=True)
