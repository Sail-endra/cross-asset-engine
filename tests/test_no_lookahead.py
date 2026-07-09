"""The no-lookahead guard.

The project's hard rule is that a signal at date t may use only information
available at or before t. These tests fail loudly if that is ever violated:

- `test_production_momentum_does_not_peek` pins the production momentum
  position builder to a hand-written honest (backward-only) implementation.
  If anyone edits momentum to reference a future row, this assertion breaks.

- `test_leaked_signal_scores_an_impossible_hit_rate` demonstrates the guard's
  teeth: a signal that *does* peek at the forward return scores a perfect,
  physically impossible hit rate, which the honest signal never does. This is
  the canary -- a real signal approaching 100% hit rate is a lookahead smell.
"""

import numpy as np
import pandas as pd

from cross_asset_engine.backtest.engine import run_backtest, forward_returns
from cross_asset_engine.backtest.strategies import momentum_strategy


def _prices():
    rng = np.random.default_rng(42)
    # random walk with drift so momentum has something real but imperfect
    steps = rng.normal(0.0005, 0.01, 800)
    return pd.DataFrame(
        {"A": 100 * np.exp(np.cumsum(steps))},
        index=pd.date_range("2015-01-01", periods=800, freq="B"),
    )


def test_production_momentum_does_not_peek():
    prices = _prices()
    lookback = 63
    positions, _ = momentum_strategy(prices, lookback)

    # Honest reference: sign of (P_t / P_{t-lb} - 1), using only past prices.
    honest = np.sign(prices / prices.shift(lookback) - 1.0)

    # They must be identical. If momentum is ever changed to use a future
    # price (e.g. prices.shift(-h)), this equality fails.
    pd.testing.assert_frame_equal(positions.fillna(0.0), honest.fillna(0.0))


def test_leaked_signal_scores_an_impossible_hit_rate():
    prices = _prices()
    horizon = 21

    # Honest momentum: past-only.
    honest_pos, _ = momentum_strategy(prices, 63)
    honest = run_backtest("honest", honest_pos, prices, horizon=horizon, cost_bps=0.0)

    # Leaked "signal": position = sign of the FORWARD return (cheating).
    leaked_pos = np.sign(forward_returns(prices, horizon))
    leaked = run_backtest("leaked", leaked_pos, prices, horizon=horizon, cost_bps=0.0)

    # The cheat is right essentially every period; the honest signal is not.
    assert leaked.metrics_gross.hit_rate > 0.99
    assert honest.metrics_gross.hit_rate < 0.90
    # And the leak's paper Sharpe is absurd relative to the honest one.
    assert leaked.metrics_gross.sharpe > honest.metrics_gross.sharpe
