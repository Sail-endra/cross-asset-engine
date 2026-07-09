import pandas as pd
import pytest

from cross_asset_engine.data.exceptions import DataFetchError, DataGapError
from cross_asset_engine.data.schema import (
    check_no_large_gaps,
    make_tidy_frame,
    validate_tidy_frame,
)


def test_make_tidy_frame_happy_path():
    df = make_tidy_frame(
        dates=["2026-01-02", "2026-01-01"],
        asset_id="10Y",
        values=["4.5", "4.4"],
        source="fred",
    )
    assert list(df.columns) == ["date", "asset_id", "value", "source"]
    # sorted ascending even though input was descending
    assert df["date"].is_monotonic_increasing
    assert df["value"].tolist() == [4.4, 4.5]
    assert (df["asset_id"] == "10Y").all()
    assert (df["source"] == "fred").all()


def test_make_tidy_frame_rejects_non_numeric_values():
    with pytest.raises(DataFetchError):
        make_tidy_frame(
            dates=["2026-01-01", "2026-01-02"],
            asset_id="10Y",
            values=["4.5", "not-a-number"],
            source="fred",
        )


def test_make_tidy_frame_drops_duplicate_dates():
    df = make_tidy_frame(
        dates=["2026-01-01", "2026-01-01"],
        asset_id="10Y",
        values=["4.5", "4.5"],
        source="fred",
    )
    assert len(df) == 1


def test_validate_tidy_frame_rejects_missing_columns():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "value": [1.0]})
    with pytest.raises(DataFetchError):
        validate_tidy_frame(df)


def test_validate_tidy_frame_rejects_empty():
    df = pd.DataFrame(columns=["date", "asset_id", "value", "source"])
    with pytest.raises(DataFetchError):
        validate_tidy_frame(df)


def test_check_no_large_gaps_passes_on_daily_data():
    df = make_tidy_frame(
        dates=["2026-01-01", "2026-01-02", "2026-01-05"],  # weekend gap
        asset_id="10Y",
        values=["4.5", "4.5", "4.5"],
        source="fred",
    )
    check_no_large_gaps(df, max_gap_calendar_days=10)


def test_check_no_large_gaps_fails_on_multiweek_gap():
    df = make_tidy_frame(
        dates=["2026-01-01", "2026-03-01"],
        asset_id="10Y",
        values=["4.5", "4.6"],
        source="fred",
    )
    with pytest.raises(DataGapError):
        check_no_large_gaps(df, max_gap_calendar_days=10)
