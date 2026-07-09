"""Shared tidy schema that every vendor loader must return.

Signals, the curve module, and the briefing all consume this one shape --
date index, asset id, value, source tag -- so they never need to know
whether a number came from FRED or Alpha Vantage. Keeping this contract in
one place is what makes it possible to swap or add a vendor without
touching any downstream code.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from .exceptions import DataFetchError, DataGapError

TIDY_COLUMNS = ["date", "asset_id", "value", "source"]


def make_tidy_frame(
    dates: Iterable,
    asset_id: str,
    values: Sequence,
    source: str,
) -> pd.DataFrame:
    """Build and validate a tidy single-asset time series.

    Raises DataFetchError (rather than silently coercing) if a value cannot
    be parsed as numeric -- a vendor payload with unexpected non-numeric
    content is a data-quality problem, not something to paper over.
    """
    try:
        numeric_values = pd.to_numeric(pd.Series(list(values)), errors="raise")
    except (ValueError, TypeError) as e:
        raise DataFetchError(
            f"Non-numeric value in series for asset_id={asset_id!r} (source={source!r}): {e}"
        ) from e

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)).normalize(),
            "asset_id": asset_id,
            "value": numeric_values.to_numpy(),
            "source": source,
        }
    )
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    validate_tidy_frame(df)
    return df


def validate_tidy_frame(df: pd.DataFrame) -> None:
    """Structural checks every tidy frame must pass before use downstream."""
    missing_cols = set(TIDY_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataFetchError(f"Tidy frame missing required columns: {missing_cols}")
    if df.empty:
        raise DataFetchError("Tidy frame has no rows")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise DataFetchError("'date' column must be datetime64")
    if df["date"].duplicated().any():
        raise DataFetchError("Tidy frame has duplicate dates for a single asset_id")
    if df["value"].isna().any():
        raise DataFetchError(
            "Tidy frame contains NaN values -- upstream loaders must drop or "
            "fail on missing observations, never pass NaN through silently"
        )


def check_no_large_gaps(df: pd.DataFrame, max_gap_calendar_days: int) -> None:
    """Fail loudly if consecutive observations are farther apart than expected.

    A one- or two-day gap is a normal weekend or holiday. A gap beyond
    max_gap_calendar_days almost always means the vendor stopped publishing
    or a request parameter was wrong -- exactly the silent-gap failure mode
    this project must not paper over.
    """
    gaps = df["date"].diff().dt.days.dropna()
    if (gaps > max_gap_calendar_days).any():
        worst = gaps.max()
        asset_id = df["asset_id"].iloc[0]
        raise DataGapError(
            f"asset_id={asset_id!r} has a {int(worst)}-day gap between observations, "
            f"exceeding the max_gap_calendar_days={max_gap_calendar_days} threshold"
        )
