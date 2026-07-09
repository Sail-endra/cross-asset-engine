import numpy as np
import pytest

from cross_asset_engine.curve.bootstrap import (
    bootstrap_zero_curve,
    implied_forward_curve,
    interpolate_par_yields,
)


def test_flat_par_curve_bootstraps_to_flat_zero_curve():
    # Analytic fact: a flat par yield y implies DF(t) = 1/(1+y/2)^(2t) and an
    # identically flat zero curve = y. This is the cleanest correctness check.
    tenors = [0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    yields = [4.0] * len(tenors)
    curve = bootstrap_zero_curve(tenors, yields, step_years=0.5, max_maturity=30)
    assert np.allclose(curve.zero_rates_pct, 4.0, atol=1e-6)
    # DF at 1y == 1/1.02^2
    assert curve.df(1.0) == pytest.approx(1.0 / (1.02 ** 2), rel=1e-9)


def test_two_node_bootstrap_matches_hand_computation():
    # 6M par = 4.0%, 1Y par = 4.5%. By hand:
    #   DF(0.5) = 1/(1+0.02) = 0.9803922
    #   DF(1)   = (1 - 0.0225*DF(0.5))/(1.0225) = 0.9564217
    #   zero(1) = 2*(DF(1)^-0.5 - 1)*100 ~= 4.506%
    curve = bootstrap_zero_curve([0.5, 1.0], [4.0, 4.5], step_years=0.5, max_maturity=1.0)
    assert curve.discount_factors[0] == pytest.approx(0.9803922, abs=1e-6)
    assert curve.discount_factors[1] == pytest.approx(0.9564217, abs=1e-6)
    assert curve.zero_rate(1.0) == pytest.approx(4.506, abs=3e-3)


def test_upward_sloping_par_gives_zero_above_par_at_long_end():
    # For an upward-sloping curve the zero rate sits above the par yield at the
    # long end (par is a blend of lower earlier zeros + the final one).
    tenors = [0.5, 1, 2, 5, 10]
    yields = [3.0, 3.3, 3.7, 4.2, 4.6]
    curve = bootstrap_zero_curve(tenors, yields, step_years=0.5, max_maturity=10)
    assert curve.zero_rate(10.0) > 4.6


def test_forward_rates_exceed_spot_when_curve_slopes_up():
    tenors = [0.5, 1, 2, 5, 10]
    yields = [3.0, 3.3, 3.7, 4.2, 4.6]
    curve = bootstrap_zero_curve(tenors, yields, step_years=0.5, max_maturity=10)
    fwd = implied_forward_curve(curve)
    # the far forward should be above the near zero on an upward-sloping curve
    assert fwd[-1] > curve.zero_rates_pct[0]


def test_interpolation_linear_is_exact_at_nodes():
    y = interpolate_par_yields([2, 10], [4.0, 5.0], [2, 6, 10], "linear")
    assert y[0] == pytest.approx(4.0)
    assert y[1] == pytest.approx(4.5)  # midpoint
    assert y[2] == pytest.approx(5.0)


def test_interpolation_clamps_outside_range():
    y = interpolate_par_yields([2, 10], [4.0, 5.0], [1, 20], "linear")
    assert y[0] == pytest.approx(4.0)  # flat below first node
    assert y[1] == pytest.approx(5.0)  # flat above last node
