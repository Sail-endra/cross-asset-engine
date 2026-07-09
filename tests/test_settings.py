from cross_asset_engine.settings import load_instruments, load_params


def test_shipped_instruments_yaml_parses_and_has_expected_shape():
    instruments = load_instruments()
    assert len(instruments["rates"]["curve"]) == 11
    assert len(instruments["credit"]) == 2
    assert len(instruments["equities"]) >= 1
    assert len(instruments["fx"]) >= 1
    tenors = {p["tenor"] for p in instruments["rates"]["curve"]}
    assert {"1M", "2Y", "10Y", "30Y"} <= tenors


def test_shipped_params_yaml_parses_and_has_expected_keys():
    params = load_params()
    assert "cache_dir" in params["data"]
    assert params["data"]["max_gap_calendar_days"] > 0
