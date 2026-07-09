"""The backtest engine: position panel + price panel -> P&L and metrics.

No-lookahead is enforced structurally here (this is the project's most
important correctness property):

    position(t)   is formed from data up to and including date t
    forward_ret(t) = price(t+h)/price(t) - 1   is realized AFTER t

The engine pairs position(t) with forward_ret(t) and never the other way
around. A position can therefore only ever earn a return that happens strictly
after the information used to form it. `backtest/strategies.py` is where the
positions are built, and `tests/test_no_lookahead.py` fails loudly if a builder
is ever changed to peek at future prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import Metrics, compute_metrics


@dataclass
class BacktestResult:
    name: str
    horizon_days: int
    periods_per_year: float
    period_returns_gross: pd.Series
    period_returns_net: pd.Series
    turnover: pd.Series
    equity_gross: pd.Series
    equity_net: pd.Series
    metrics_gross: Metrics
    metrics_net: Metrics
    units: str = "return"  # "return" for %/decimal strategies, "bp" for curve trades


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return over `horizon` rows: price(t+h)/price(t) - 1, indexed at t."""
    return prices.shift(-horizon) / prices - 1.0


def _normalize_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """Scale each row's +/-1/0 positions to unit gross exposure (sum|w| = 1)."""
    gross = positions.abs().sum(axis=1)
    gross = gross.replace(0.0, np.nan)
    return positions.div(gross, axis=0).fillna(0.0)


def run_backtest(
    name: str,
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    cost_bps: float = 0.0,
    risk_free_pct: float = 0.0,
    units: str = "return",
) -> BacktestResult:
    """Backtest a position panel against a price panel, non-overlapping by horizon.

    positions and prices are date-indexed, same columns (assets). Positions are
    sampled every `horizon` rows so periods don't overlap; each sampled position
    earns the forward return over the following `horizon` days, minus costs on
    the traded notional.
    """
    cols = [c for c in positions.columns if c in prices.columns]
    positions = positions[cols].reindex(prices.index).fillna(0.0)
    fwd = forward_returns(prices[cols], horizon)

    weights = _normalize_weights(positions)

    # Non-overlapping rebalance rows: 0, h, 2h, ...
    idx = np.arange(0, len(prices.index), horizon)
    rebal_dates = prices.index[idx]

    gross_rets, turnovers = [], []
    prev_w = pd.Series(0.0, index=cols)
    kept_dates = []
    for d in rebal_dates:
        w = weights.loc[d]
        fr = fwd.loc[d]
        if fr.isna().all():
            continue  # no full forward window (end of sample)
        # portfolio return over the horizon; assets without a fwd return sit out
        contrib = (w * fr).dropna()
        gross = float(contrib.sum())
        traded = float((w - prev_w).abs().sum())  # notional traded this rebalance
        gross_rets.append(gross)
        turnovers.append(traded)
        kept_dates.append(d)
        prev_w = w

    gross_series = pd.Series(gross_rets, index=pd.DatetimeIndex(kept_dates), name="gross")
    turnover_series = pd.Series(turnovers, index=pd.DatetimeIndex(kept_dates), name="turnover")
    costs = turnover_series * (cost_bps / 1e4)
    net_series = (gross_series - costs).rename("net")

    periods_per_year = 252.0 / horizon
    equity_gross = (1.0 + gross_series).cumprod()
    equity_net = (1.0 + net_series).cumprod()

    return BacktestResult(
        name=name,
        horizon_days=horizon,
        periods_per_year=periods_per_year,
        period_returns_gross=gross_series,
        period_returns_net=net_series,
        turnover=turnover_series,
        equity_gross=equity_gross,
        equity_net=equity_net,
        metrics_gross=compute_metrics(gross_series, periods_per_year, turnover_series, risk_free_pct),
        metrics_net=compute_metrics(net_series, periods_per_year, turnover_series, risk_free_pct),
        units=units,
    )
