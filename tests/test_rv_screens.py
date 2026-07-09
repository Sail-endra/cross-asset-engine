import numpy as np
import pandas as pd
import pytest

from cross_asset_engine.curve.rv_screens import (
    butterfly_screen,
    carry_rolldown_screen,
    latest_yields_from_panel,
    rolldown_for_carry,
    slope_screen,
)


def _yield_panel(latest: dict, n=300, jitter=0.0):
    """Build a date-indexed yield panel that is flat history then `latest`."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for tenor, val in latest.items():
        col = np.full(n, val - jitter)
        col[-1] = val
        data[tenor] = col
    return pd.DataFrame(data, index=idx)


def test_slope_bp_and_dv01_neutral_sizing():
    panel = _yield_panel({"2Y": 4.0, "10Y": 4.5})
    res = slope_screen(panel, "2s10s", "2Y", "10Y", window=252)
    assert res.slope_bp == pytest.approx((4.5 - 4.0) * 100.0)  # 50 bp
    # long leg (10Y) has more DV01 than short leg (2Y)
    assert res.dv01_long > res.dv01_short
    # DV01-neutral: hold >1 unit of the 2Y per unit of 10Y
    assert res.hedge_ratio_short_per_long > 1.0


def test_slope_zscore_flags_steepening_as_flattener():
    # slope jumps to a historic high on the last day -> z large positive.
    panel = _yield_panel({"2Y": 4.0, "10Y": 4.5}, jitter=0.15)  # history slope=35bp, latest=50bp
    res = slope_screen(panel, "2s10s", "2Y", "10Y", window=252)
    assert res.zscore > 1.0
    assert "flattener" in res.signal


def test_butterfly_bp_and_wing_sizing():
    panel = _yield_panel({"2Y": 4.0, "5Y": 4.3, "10Y": 4.5})
    res = butterfly_screen(panel, "2s5s10s", "2Y", "5Y", "10Y", window=252)
    expected = (2 * 4.3 - 4.0 - 4.5) * 100.0
    assert res.fly_bp == pytest.approx(expected)
    assert res.wing_short_units_per_belly > 0
    assert res.wing_long_units_per_belly > 0


def test_carry_rolldown_positive_on_upward_sloping_curve():
    # Include front tenors so points other than the very shortest have curve
    # to roll down into. The shortest point (1Y here) rolls below the curve, so
    # its rolldown clamps to 0 -- correct "no roll info" behavior, not a bug.
    latest = {"1Y": 3.7, "2Y": 4.0, "3Y": 4.15, "5Y": 4.3, "7Y": 4.4, "10Y": 4.5}
    points = carry_rolldown_screen(latest, horizon_years=0.25)
    assert len(points) > 0
    for p in points:
        assert p.roll_yield_bp >= 0  # upward slope -> non-negative roll
    # everything above the shortest tenor genuinely rolls down
    non_shortest = [p for p in points if p.tenor != "1Y"]
    assert all(p.rolldown_return_pct > 0 for p in non_shortest)
    # ranked by total-per-DV01 descending
    tpd = [p.total_per_dv01 for p in points]
    assert tpd == sorted(tpd, reverse=True)


def test_rolldown_for_carry_returns_annualized_nonneg_values():
    latest = {"1Y": 3.7, "2Y": 4.0, "5Y": 4.3, "10Y": 4.5}
    rd = rolldown_for_carry(latest, horizon_years=0.25)
    assert all(v >= 0 for v in rd.values())
    assert any(v > 0 for v in rd.values())


def test_latest_yields_from_panel_drops_nans():
    panel = _yield_panel({"2Y": 4.0, "10Y": 4.5})
    panel.loc[panel.index[-1], "2Y"] = np.nan
    latest = latest_yields_from_panel(panel)
    assert "2Y" not in latest
    assert latest["10Y"] == pytest.approx(4.5)
