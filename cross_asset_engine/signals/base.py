"""Shared plumbing for the signal engine.

Every signal -- momentum, carry, regime, and any added later (e.g. the
Perihelion volatility-risk-premium signal) -- emits the same standardized
`SignalResult`, and registers itself in `SIGNAL_REGISTRY`. That uniformity
is what lets the backtester and the briefing iterate over signals generically
without special-casing each one.

Two helpers here matter for the project's no-lookahead rule:

- `pivot_to_panel` turns a tidy (date, asset_id, value) frame into a wide
  date x asset panel, aligned on a common date index. All signal maths runs
  on these panels, where "a row is one day" makes it obvious that a trailing
  window at date t only ever reaches backwards.
- `trailing_return` is deliberately written so the value at row t uses only
  rows <= t; there is no centering or forward window anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    """One signal's read on one asset on one date.

    Attributes
    ----------
    signal : str
        Family name, e.g. "momentum", "carry", "regime".
    variant : str
        The specific configuration, e.g. "ts_126d" or "cross_sectional_252d".
    asset_id : str
        Which instrument (e.g. "SP500", "EURUSD", "10Y", "US_HY_OAS").
    asset_class : str
        "rates" | "credit" | "equities" | "fx".
    date : pd.Timestamp
        The date the signal is computed *as of* (all inputs are <= this date).
    score : float
        Continuous signal strength (e.g. trailing return, carry in %,
        cross-sectional z-score). Sign convention: positive = bullish the
        asset / long the risk.
    direction : int
        Discretized position: +1 long, -1 short, 0 flat.
    inputs : dict
        The raw numbers the signal was built from, kept so every value in the
        briefing is traceable back to its drivers.
    """

    signal: str
    variant: str
    asset_id: str
    asset_class: str
    date: pd.Timestamp
    score: float
    direction: int
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.strftime("%Y-%m-%d")
        d["score"] = None if self.score is None or np.isnan(self.score) else float(self.score)
        return d


def pivot_to_panel(tidy: pd.DataFrame) -> pd.DataFrame:
    """Tidy (date, asset_id, value) -> wide panel indexed by date.

    Columns are asset_ids, values are the series level. The union of dates is
    used; missing cells stay NaN rather than being filled, so downstream code
    decides explicitly how to handle them (never a silent fill here).
    """
    panel = tidy.pivot_table(index="date", columns="asset_id", values="value", aggfunc="last")
    return panel.sort_index()


def trailing_return(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Simple trailing return over `lookback` rows, strictly backward-looking.

    r_t = P_t / P_{t-lookback} - 1, using only prices up to and including t.
    Returns NaN for the first `lookback` rows where no full window exists.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    return panel / panel.shift(lookback) - 1.0


def cross_sectional_zscore(row: pd.Series) -> pd.Series:
    """Demean and scale a cross-section to z-scores (favor the strongest).

    Used by cross-sectional signals: within one date, rank assets against
    each other. NaNs (assets without enough history) are ignored.
    """
    valid = row.dropna()
    if len(valid) < 2 or valid.std(ddof=1) == 0:
        return pd.Series(np.nan, index=row.index)
    z = (row - valid.mean()) / valid.std(ddof=1)
    return z


# ---- Registry -------------------------------------------------------------

SIGNAL_REGISTRY: Dict[str, Callable] = {}


def register_signal(name: str) -> Callable:
    """Decorator: register a signal factory under `name` for generic iteration."""

    def _wrap(fn: Callable) -> Callable:
        if name in SIGNAL_REGISTRY:
            raise ValueError(f"Signal {name!r} is already registered")
        SIGNAL_REGISTRY[name] = fn
        return fn

    return _wrap


def list_signals() -> list[str]:
    return sorted(SIGNAL_REGISTRY)
