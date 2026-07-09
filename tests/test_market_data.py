from datetime import date

import pytest
import yaml

from cross_asset_engine.data.alphavantage import AlphaVantageLoader
from cross_asset_engine.data.cache import RawCache
from cross_asset_engine.data.exceptions import DataGapError
from cross_asset_engine.data.fred import FredLoader
from cross_asset_engine.data.market_data import MarketData

from .conftest import FakeResponse, FakeSession


@pytest.fixture
def small_instruments(tmp_path):
    instruments = {
        "rates": {"curve": [{"tenor": "2Y", "series_id": "DGS2"}, {"tenor": "10Y", "series_id": "DGS10"}]},
        "credit": [{"asset_id": "US_IG_OAS", "series_id": "BAMLC0A0CM", "label": "IG OAS"}],
        "equities": [{"asset_id": "SP500", "series_id": "SP500", "label": "S&P 500"}],
        "fx": [{"asset_id": "EURUSD", "from_symbol": "EUR", "to_symbol": "USD", "label": "EURUSD"}],
    }
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump(instruments))
    return path


@pytest.fixture
def params_file(tmp_path):
    params = {
        "data": {
            "cache_dir": "cache",
            "fred_observation_start": "2020-01-01",
            "max_gap_calendar_days": 10,
        }
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path


def _fred_payload(values_by_date):
    return {"observations": [{"date": d, "value": v} for d, v in values_by_date.items()]}


def make_market_data(tmp_path, small_instruments, params_file, fred_responses, av_responses):
    cache = RawCache(tmp_path / "cache")
    fred = FredLoader(cache, api_key="fake", session=FakeSession(fred_responses))
    av = AlphaVantageLoader(cache, api_key="fake", session=FakeSession(av_responses))
    return MarketData(
        instruments_path=small_instruments,
        params_path=params_file,
        as_of=date(2026, 1, 6),
        project_root=tmp_path,
        fred_loader=fred,
        av_loader=av,
    )


def test_get_all_concatenates_every_asset_class(tmp_path, small_instruments, params_file):
    fred_responses = [
        FakeResponse(_fred_payload({"2026-01-01": "4.0", "2026-01-02": "4.1"})),  # 2Y
        FakeResponse(_fred_payload({"2026-01-01": "4.5", "2026-01-02": "4.6"})),  # 10Y
        FakeResponse(_fred_payload({"2026-01-01": "100", "2026-01-02": "101"})),  # IG OAS
        FakeResponse(_fred_payload({"2026-01-01": "4000", "2026-01-02": "4050"})),  # SP500
    ]
    av_responses = [
        FakeResponse({"Time Series FX (Daily)": {"2026-01-01": {"4. close": "1.10"}, "2026-01-02": {"4. close": "1.11"}}}),
    ]
    md = make_market_data(tmp_path, small_instruments, params_file, fred_responses, av_responses)

    all_data = md.get_all()

    assert set(all_data["asset_class"].unique()) == {"rates", "credit", "equities", "fx"}
    assert set(all_data["asset_id"].unique()) == {"2Y", "10Y", "US_IG_OAS", "SP500", "EURUSD"}


def test_latest_levels_and_changes_computes_day_over_day_delta(tmp_path, small_instruments, params_file):
    fred_responses = [
        FakeResponse(_fred_payload({"2026-01-01": "4.0", "2026-01-02": "4.1"})),
        FakeResponse(_fred_payload({"2026-01-01": "4.5", "2026-01-02": "4.6"})),
        FakeResponse(_fred_payload({"2026-01-01": "100", "2026-01-02": "101"})),
        FakeResponse(_fred_payload({"2026-01-01": "4000", "2026-01-02": "4050"})),  # SP500
    ]
    av_responses = [
        FakeResponse({"Time Series FX (Daily)": {"2026-01-01": {"4. close": "1.10"}, "2026-01-02": {"4. close": "1.11"}}}),
    ]
    md = make_market_data(tmp_path, small_instruments, params_file, fred_responses, av_responses)

    summary = md.latest_levels_and_changes()
    row = summary.set_index("asset_id").loc["2Y"]

    assert row["value"] == pytest.approx(4.1)
    assert row["change"] == pytest.approx(0.1)


def test_large_gap_in_one_series_fails_the_whole_pull(tmp_path, small_instruments, params_file):
    fred_responses = [
        FakeResponse(_fred_payload({"2026-01-01": "4.0", "2026-03-01": "4.9"})),  # huge gap
        FakeResponse(_fred_payload({"2026-01-01": "4.5", "2026-01-02": "4.6"})),
        FakeResponse(_fred_payload({"2026-01-01": "100", "2026-01-02": "101"})),
        FakeResponse(_fred_payload({"2026-01-01": "4000", "2026-01-02": "4050"})),  # SP500
    ]
    av_responses = [
        FakeResponse({"Time Series FX (Daily)": {"2026-01-01": {"4. close": "1.10"}, "2026-01-02": {"4. close": "1.11"}}}),
    ]
    md = make_market_data(tmp_path, small_instruments, params_file, fred_responses, av_responses)

    with pytest.raises(DataGapError):
        md.get_all()
