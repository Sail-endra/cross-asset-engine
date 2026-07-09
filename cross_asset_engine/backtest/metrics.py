"""Performance metrics for a strategy's per-period return stream.

Finance rationale
-----------------
These are the numbers an interviewer will ask for: does the signal actually
make money, how much risk did it take, and how often was it right. Everything
is computed from a series of *non-overlapping* period returns (the strategy is
sampled every `horizon` days), so the periods are independent enough that the
annualization and the Sharpe ratio are honest rather than inflated by
overlapping-window autocorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    n_periods: int
    hit_rate: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float

    def to_dict(self) -> Dict[str, float]:
        return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                for k, v in asdict(self).items()}


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve (a negative number)."""
    if len(equity) == 0:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_metrics(
    period_returns: pd.Series,
    periods_per_year: float,
    turnover: Optional[pd.Series] = None,
    risk_free_pct: float = 0.0,
) -> Metrics:
    """Turn a per-period return series into the standard metric set.

    hit_rate    -- fraction of periods with a positive return
    ann_return  -- mean period return * periods_per_year
    ann_vol     -- period stdev * sqrt(periods_per_year)
    sharpe      -- (ann_return - rf) / ann_vol
    max_dd      -- from the compounded equity curve
    avg_turnover-- mean traded notional per rebalance (0 = never trades)
    """
    r = period_returns.dropna()
    n = len(r)
    if n == 0:
        return Metrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    hit = float((r > 0).mean())
    ann_ret = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")
    rf = risk_free_pct / 100.0
    sharpe = float((ann_ret - rf) / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod()
    mdd = max_drawdown(equity)
    avg_to = float(turnover.dropna().mean()) if turnover is not None and len(turnover.dropna()) else 0.0
    return Metrics(n, hit, ann_ret, ann_vol, sharpe, mdd, avg_to)
