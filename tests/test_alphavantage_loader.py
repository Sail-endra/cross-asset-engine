from datetime import date

import pytest

from cross_asset_engine.data.alphavantage import AlphaVantageLoader
from cross_asset_engine.data.exceptions import DataFetchError

from .conftest import FakeResponse, FakeSession


def test_fetch_equity_daily_happy_path(cache):
    payload = {
        "Time Series (Daily)": {
            "2026-01-02": {"4. close": "101.0"},
            "2026-01-01": {"4. close": "100.0"},
        }
    }
    session = FakeSession([FakeResponse(payload)])
    loader = AlphaVantageLoader(cache, api_key="fakekey", session=session)

    df = loader.fetch_equity_daily("SPY", "SPY", as_of=date(2026, 1, 3))

    assert df["value"].tolist() == [100.0, 101.0]
    assert len(session.calls) == 1


def test_fetch_fx_daily_happy_path(cache):
    payload = {
        "Time Series FX (Daily)": {
            "2026-01-01": {"4. close": "1.10"},
        }
    }
    session = FakeSession([FakeResponse(payload)])
    loader = AlphaVantageLoader(cache, api_key="fakekey", session=session)

    df = loader.fetch_fx_daily("EUR", "USD", "EURUSD", as_of=date(2026, 1, 3))

    assert df["value"].tolist() == [1.10]


def test_second_call_same_day_uses_cache_not_new_request(cache):
    payload = {"Time Series (Daily)": {"2026-01-01": {"4. close": "100.0"}}}
    session = FakeSession([FakeResponse(payload)])
    loader = AlphaVantageLoader(cache, api_key="fakekey", session=session)

    loader.fetch_equity_daily("SPY", "SPY", as_of=date(2026, 1, 3))
    loader.fetch_equity_daily("SPY", "SPY", as_of=date(2026, 1, 3))

    assert len(session.calls) == 1  # quota-preserving cache hit on 2nd call


def test_rate_limit_note_raises_instead_of_silently_passing(cache):
    payload = {"Information": "You have exceeded the daily rate limit."}
    session = FakeSession([FakeResponse(payload)])
    loader = AlphaVantageLoader(cache, api_key="fakekey", session=session)

    with pytest.raises(DataFetchError):
        loader.fetch_equity_daily("SPY", "SPY", as_of=date(2026, 1, 3))


def test_error_message_raises(cache):
    payload = {"Error Message": "Invalid API call."}
    session = FakeSession([FakeResponse(payload)])
    loader = AlphaVantageLoader(cache, api_key="fakekey", session=session)

    with pytest.raises(DataFetchError):
        loader.fetch_equity_daily("BADSYM", "BADSYM", as_of=date(2026, 1, 3))
