import numpy as np
import pandas as pd

from cross_asset_engine.signals.base import trailing_return, pivot_to_panel
from cross_asset_engine.signals.momentum import (
    par_bond_modified_duration,
    yields_to_total_return_index,
    time_series_signals,
    cross_sectional_signals,
)


def _panel(series: dict, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(next(iter(series.values()))), freq="B")
    return pd.DataFrame(series, index=idx)


def test_trailing_return_is_backward_looking_and_correct():
    panel = _panel({"A": [100, 110, 121]})  # +10% each step
    r = trailing_return(panel, 1)
    assert np.isnan(r["A"].iloc[0])  # no window yet
    assert r["A"].iloc[1] == pytest_approx(0.10)
    assert r["A"].iloc[2] == pytest_approx(0.10)


def test_time_series_momentum_sign_up_and_down():
    up = list(np.linspace(100, 150, 300))
    down = list(np.linspace(150, 100, 300))
    panel = _panel({"UP": up, "DOWN": down})
    sigs = time_series_signals(panel, "equities", [126], as_of=panel.index[-1])
    by_asset = {s.asset_id: s for s in sigs}
    assert by_asset["UP"].direction == 1
    assert by_asset["DOWN"].direction == -1


def test_cross_sectional_favors_strongest():
    strong = list(np.linspace(100, 200, 300))
    weak = list(np.linspace(100, 105, 300))
    panel = _panel({"STRONG": strong, "WEAK": weak})
    sigs = cross_sectional_signals(panel, "equities", [126], as_of=panel.index[-1])
    by_asset = {s.asset_id: s for s in sigs}
    # strongest gets a positive z, weakest negative
    assert by_asset["STRONG"].score > 0
    assert by_asset["WEAK"].score < 0
    assert by_asset["STRONG"].direction == 1
    assert by_asset["WEAK"].direction == -1


def test_par_bond_duration_reasonable():
    # A 10y par bond around 4% should have modified duration ~8 years.
    d = par_bond_modified_duration(4.0, 10.0)
    assert 7.0 < d < 9.0
    # A 3-month bill has tiny duration.
    assert par_bond_modified_duration(4.0, 0.25) < 0.3


def test_falling_yields_produce_positive_bond_momentum():
    # Yields fall 5.0 -> 3.0 over the window: bond TR index must RISE, so
    # momentum on the TR proxy must be positive (long duration). This is the
    # sign-convention guard: naive momentum on the yield level would be negative.
    yields = _panel({"10Y": list(np.linspace(5.0, 3.0, 300))})
    tr = yields_to_total_return_index(yields)
    assert tr["10Y"].iloc[-1] > tr["10Y"].iloc[0]
    sigs = time_series_signals(tr, "rates", [252], as_of=tr.index[-1])
    assert sigs[0].direction == 1


def test_rising_yields_produce_negative_bond_momentum():
    yields = _panel({"10Y": list(np.linspace(3.0, 5.0, 300))})
    tr = yields_to_total_return_index(yields)
    assert tr["10Y"].iloc[-1] < tr["10Y"].iloc[0]
    sigs = time_series_signals(tr, "rates", [252], as_of=tr.index[-1])
    assert sigs[0].direction == -1


def pytest_approx(x, rel=1e-6):
    import pytest

    return pytest.approx(x, rel=rel)
