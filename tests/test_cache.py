from datetime import date

from cross_asset_engine.data.cache import RawCache


def test_cache_miss_returns_none(tmp_path):
    cache = RawCache(tmp_path)
    assert cache.get("fred", "DGS10_1990-01-01", date(2026, 1, 1)) is None


def test_cache_put_then_get_roundtrips(tmp_path):
    cache = RawCache(tmp_path)
    payload = {"observations": [{"date": "2026-01-01", "value": "4.5"}]}
    cache.put("fred", "DGS10_1990-01-01", date(2026, 1, 1), payload)
    assert cache.get("fred", "DGS10_1990-01-01", date(2026, 1, 1)) == payload


def test_cache_is_dated_not_shared_across_days(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("fred", "DGS10_1990-01-01", date(2026, 1, 1), {"v": 1})
    # a different as_of date is a cache miss even for the same key
    assert cache.get("fred", "DGS10_1990-01-01", date(2026, 1, 2)) is None


def test_cache_keys_with_slashes_are_sanitized(tmp_path):
    cache = RawCache(tmp_path)
    cache.put("alpha_vantage", "fx_EUR/USD", date(2026, 1, 1), {"v": 1})
    assert cache.get("alpha_vantage", "fx_EUR/USD", date(2026, 1, 1)) == {"v": 1}
