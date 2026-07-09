from datetime import date

import pytest

from cross_asset_engine.data.exceptions import DataFetchError
from cross_asset_engine.data.fred import FredLoader

from .conftest import FakeResponse, FakeSession


def test_fetch_series_happy_path(cache):
    payload = {
        "observations": [
            {"date": "2026-01-01", "value": "4.50"},
            {"date": "2026-01-02", "value": "."},  # holiday marker, must be dropped
            {"date": "2026-01-05", "value": "4.55"},
        ]
    }
    session = FakeSession([FakeResponse(payload)])
    loader = FredLoader(cache, api_key="fakekey", session=session)

    df = loader.fetch_series("DGS10", "10Y", as_of=date(2026, 1, 6))

    assert len(df) == 2  # the "." row was dropped
    assert df["value"].tolist() == [4.50, 4.55]
    assert len(session.calls) == 1


def test_fetch_series_uses_cache_on_second_call(cache):
    payload = {"observations": [{"date": "2026-01-01", "value": "4.50"}]}
    session = FakeSession([FakeResponse(payload)])
    loader = FredLoader(cache, api_key="fakekey", session=session)

    loader.fetch_series("DGS10", "10Y", as_of=date(2026, 1, 6))
    loader.fetch_series("DGS10", "10Y", as_of=date(2026, 1, 6))  # should hit cache, not session

    assert len(session.calls) == 1


def test_fetch_series_raises_on_error_payload(cache):
    payload = {"error_code": 400, "error_message": "Bad Request."}
    session = FakeSession([FakeResponse(payload)])
    loader = FredLoader(cache, api_key="fakekey", session=session)

    with pytest.raises(DataFetchError):
        loader.fetch_series("DGS10", "10Y", as_of=date(2026, 1, 6))


def test_fetch_series_raises_on_all_missing_observations(cache):
    payload = {"observations": [{"date": "2026-01-01", "value": "."}]}
    session = FakeSession([FakeResponse(payload)])
    loader = FredLoader(cache, api_key="fakekey", session=session)

    with pytest.raises(DataFetchError):
        loader.fetch_series("DGS10", "10Y", as_of=date(2026, 1, 6))


def test_missing_api_key_fails_loudly(cache, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(DataFetchError):
        FredLoader(cache, api_key=None, session=FakeSession([]))
