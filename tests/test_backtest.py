import numpy as np
import pandas as pd
import pytest

from cross_asset_engine.backtest.engine import run_backtest, forward_returns
from cross_asset_engine.backtest.metrics import compute_metrics, max_drawdown


def _price_panel(series, start="2015-01-01"):
    idx = pd.date_range(start, periods=len(next(iter(series.values()))), freq="B")
    return pd.DataFrame(series, index=idx)


def test_forward_returns_are_forward_looking():
    px = _price_panel({"A": [100, 110, 121, 133.1]})
    fwd = forward_returns(px, 1)
    assert fwd["A"].iloc[0] == pytest.approx(0.10)   # 100 -> 110
    assert np.isnan(fwd["A"].iloc[-1])               # no future row


def test_always_long_a_rising_asset_makes_money():
    prices = _price_panel({"A": list(np.linspace(100, 200, 260))})
    positions = pd.DataFrame(1.0, index=prices.index, columns=["A"])
    res = run_backtest("long", positions, prices, horizon=21, cost_bps=0.0)
    assert res.metrics_gross.ann_return > 0
    assert res.metrics_gross.hit_rate > 0.9
    assert res.equity_gross.iloc[-1] > 1.0


def test_costs_reduce_net_below_gross_when_turnover_positive():
    # Alternate long/short every day so turnover is high.
    n = 260
    prices = _price_panel({"A": list(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n)))})
    pos = pd.DataFrame({"A": [1.0 if i % 2 == 0 else -1.0 for i in range(n)]}, index=prices.index)
    res = run_backtest("flip", pos, prices, horizon=5, cost_bps=10.0)
    assert res.turnover.sum() > 0
    assert res.metrics_net.ann_return < res.metrics_gross.ann_return


def test_max_drawdown_is_negative_for_a_decline():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(eq) == pytest.approx(0.9 / 1.2 - 1.0)


def test_metrics_on_known_series():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01])
    m = compute_metrics(r, periods_per_year=12)
    assert m.n_periods == 5
    assert m.hit_rate == pytest.approx(3 / 5)  # three positive periods
    assert m.ann_return == pytest.approx(r.mean() * 12)
