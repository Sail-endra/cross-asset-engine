import pytest

from cross_asset_engine.curve.bootstrap import bootstrap_zero_curve
from cross_asset_engine.curve.dv01 import (
    dv01_from_cashflows,
    dv01_par_bond,
    modified_duration_par_bond,
    par_bond_price,
)


def test_par_bond_prices_to_par_when_coupon_equals_yield():
    assert par_bond_price(4.0, 4.0, 10.0) == pytest.approx(100.0, abs=1e-6)


def test_modified_duration_10y_4pct_is_about_8():
    d = modified_duration_par_bond(4.0, 10.0)
    assert 7.8 < d < 8.3


def test_dv01_matches_duration_times_price_rule():
    # DV01 ~= ModDur * Price * 1bp. At par, Price=100.
    d = modified_duration_par_bond(4.0, 10.0)
    approx = d * 100.0 * 1e-4
    assert dv01_par_bond(4.0, 10.0) == pytest.approx(approx, rel=1e-3)


def test_dv01_increases_with_maturity():
    assert dv01_par_bond(4.0, 2.0) < dv01_par_bond(4.0, 10.0) < dv01_par_bond(4.0, 30.0)


def test_dv01_from_cashflows_positive_and_grows_with_maturity():
    curve = bootstrap_zero_curve([0.5, 1, 2, 5, 10, 30], [4.0] * 6, max_maturity=30)
    dv_2y = dv01_from_cashflows([2.0], [100.0], curve)
    dv_10y = dv01_from_cashflows([10.0], [100.0], curve)
    assert dv_2y > 0
    assert dv_10y > dv_2y
