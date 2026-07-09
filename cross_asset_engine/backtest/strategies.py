"""Strategy adapters: turn each signal into a (positions, prices) pair the
engine can backtest over full history.

Each builder returns positions formed with data up to t only. The engine pairs
them with forward returns realized after t, so lookahead is impossible by
construction as long as these builders never reference future rows -- which the
no-lookahead test verifies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..signals.base import pivot_to_panel, trailing_return
from ..signals.momentum import yields_to_total_return_index


def momentum_strategy(prices: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series momentum: position = sign of the trailing return.

    trailing_return is strictly backward-looking, so position(t) uses only
    prices up to t. Returns (positions, prices) ready for the engine.
    """
    scores = trailing_return(prices, lookback)
    positions = np.sign(scores)
    return positions, prices


def momentum_strategy_equities(market_data, lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = pivot_to_panel(market_data.get_equities())
    return momentum_strategy(prices, lookback)


def momentum_strategy_rates(market_data, lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yields = pivot_to_panel(market_data.get_curve())
    tr = yields_to_total_return_index(yields)
    return momentum_strategy(tr, lookback)


def fx_carry_strategy(market_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """FX carry: long the higher-yielding currency of each pair.

    position(t) = sign(r_base - r_quote) using policy rates known at t
    (forward-filled onto the FX calendar), earning the pair's spot return.
    """
    fx_prices = pivot_to_panel(market_data.get_fx())
    rates = pivot_to_panel(market_data.get_short_rates()).reindex(fx_prices.index).ffill()

    positions = pd.DataFrame(0.0, index=fx_prices.index, columns=fx_prices.columns)
    for item in market_data.instruments["fx"]:
        pair, base, quote = item["asset_id"], item["from_symbol"], item["to_symbol"]
        if base in rates.columns and quote in rates.columns and pair in positions.columns:
            positions[pair] = np.sign(rates[base] - rates[quote])
    return positions, fx_prices


def curve_slope_strategy(
    yield_panel: pd.DataFrame, short: str, long: str, window: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """DV01-neutral slope trade, positioned by mean-reversion of the z-score.

    The trade P&L is expressed as a return normalized per unit DV01: a synthetic
    price = exp(slope_bp / 1e4) makes the engine's forward return approximately
    equal to the slope change in bp / 1e4. Position = -sign(rolling z-score):
    fade a historically steep curve (flattener) and vice versa. The rolling
    z-score uses only trailing data, so positions are causal.
    """
    slope_bp = (yield_panel[long] - yield_panel[short]) * 100.0
    roll_mean = slope_bp.rolling(window, min_periods=window // 2).mean()
    roll_std = slope_bp.rolling(window, min_periods=window // 2).std(ddof=1)
    z = (slope_bp - roll_mean) / roll_std

    name = f"{short}{long}_slope"
    positions = pd.DataFrame({name: -np.sign(z)}, index=yield_panel.index)
    synth_price = pd.DataFrame({name: np.exp(slope_bp / 1e4)}, index=yield_panel.index)
    return positions, synth_price


def build_default_backtests_inputs(market_data, params: Dict) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Assemble the (positions, prices, units) inputs for the headline set:
    equity momentum, FX carry, and a 2s10s curve trade."""
    lb = params["backtest"]["momentum_lookback"]
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, str]] = {}

    pos, px = momentum_strategy_equities(market_data, lb)
    out[f"Equity momentum ({lb}d)"] = (pos, px, "return")

    pos, px = fx_carry_strategy(market_data)
    out["FX carry"] = (pos, px, "return")

    yields = pivot_to_panel(market_data.get_curve())
    window = params["curve"]["zscore_window"]
    pos, px = curve_slope_strategy(yields, "2Y", "10Y", window)
    out["Curve 2s10s (RV)"] = (pos, px, "return")

    return out
