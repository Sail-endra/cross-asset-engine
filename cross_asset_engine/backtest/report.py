"""Assemble backtest results into a metrics table, an equity-curve chart, and
a JSON-serializable snapshot. The chart is rendered server-side to a base64 PNG
so Phase 5 can embed it inline in a fully portable HTML briefing.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .engine import BacktestResult


def metrics_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """One row per strategy, gross and net metrics side by side."""
    rows: List[dict] = []
    for name, res in results.items():
        g, n = res.metrics_gross, res.metrics_net
        rows.append({
            "strategy": name,
            "n": g.n_periods,
            "hit%": round(100 * g.hit_rate, 1),
            "ann_ret%_gross": round(100 * g.ann_return, 2),
            "ann_ret%_net": round(100 * n.ann_return, 2),
            "ann_vol%": round(100 * g.ann_vol, 2),
            "sharpe_gross": round(g.sharpe, 2),
            "sharpe_net": round(n.sharpe, 2),
            "maxDD%": round(100 * g.max_drawdown, 1),
            "turnover": round(g.avg_turnover, 2),
        })
    return pd.DataFrame(rows)


def equity_curve_png_base64(results: Dict[str, BacktestResult], net: bool = True) -> str:
    """Render all strategies' equity curves to a base64-encoded PNG string."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    for name, res in results.items():
        eq = res.equity_net if net else res.equity_gross
        if len(eq):
            ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_title(f"Backtest equity curves ({'net' if net else 'gross'} of costs)")
    ax.set_ylabel("Growth of 1 unit")
    ax.axhline(1.0, color="#999", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def results_to_snapshot(results: Dict[str, BacktestResult]) -> dict:
    """Machine-readable snapshot of every backtest number (for the JSON audit)."""
    return {
        name: {
            "horizon_days": res.horizon_days,
            "periods_per_year": res.periods_per_year,
            "gross": res.metrics_gross.to_dict(),
            "net": res.metrics_net.to_dict(),
            "final_equity_net": float(res.equity_net.iloc[-1]) if len(res.equity_net) else None,
        }
        for name, res in results.items()
    }


def momentum_lookback_sweep(sweep: Dict[int, BacktestResult]) -> pd.DataFrame:
    """Report the spread of outcomes across lookbacks -- anti-overfitting.

    Shows every lookback's Sharpe rather than cherry-picking the best cell.
    """
    rows = [{
        "lookback_days": lb,
        "hit%": round(100 * res.metrics_gross.hit_rate, 1),
        "ann_ret%_net": round(100 * res.metrics_net.ann_return, 2),
        "sharpe_net": round(res.metrics_net.sharpe, 2),
        "maxDD%": round(100 * res.metrics_gross.max_drawdown, 1),
    } for lb, res in sorted(sweep.items())]
    return pd.DataFrame(rows)
