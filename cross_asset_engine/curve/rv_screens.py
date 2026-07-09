"""Relative-value screens on the Treasury curve: slope, butterfly, carry-roll.

Finance rationale
-----------------
The curve is rarely mispriced at the *level* -- that's a macro call. The
repeatable edge is in its *shape*: is the curve too steep or too flat, too
humped or too flat in the belly, and which points pay you the most to hold.
These three screens isolate exactly those, each sized DV01-neutral so the
trade expresses shape rather than a hidden duration bet:

- Slope (2s10s, 5s30s): long-minus-short yield. A steepener profits when the
  gap widens. Z-scored against its own history to flag rich/cheap.
- Butterfly (2s5s10s): 2*belly - wings. Isolates curvature -- is the belly
  cheap or rich relative to the wings. Z-scored likewise.
- Carry-and-rolldown: for each point, the running yield earned over a horizon
  plus the price gain from rolling down an upward-sloping curve as the bond
  ages, ranked per unit of DV01 (rate risk). This is the "what do I get paid
  to own" screen, and its rolldown feeds back into the Phase 2 rates carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..tenors import TENOR_YEARS, TENOR_ORDER
from .bootstrap import interpolate_par_yields
from .dv01 import dv01_par_bond, modified_duration_par_bond


def _zscore_latest(series: pd.Series, window: int) -> float:
    """Z-score of the most recent value vs the trailing `window` observations."""
    s = series.dropna()
    if len(s) < max(20, window // 4):
        return float("nan")
    tail = s.iloc[-window:]
    sd = tail.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((s.iloc[-1] - tail.mean()) / sd)


@dataclass
class SlopeResult:
    name: str
    short: str
    long: str
    slope_bp: float
    zscore: float
    signal: str
    dv01_short: float
    dv01_long: float
    hedge_ratio_short_per_long: float  # units of short leg per 1 unit long leg
    date: pd.Timestamp


def slope_screen(yield_panel: pd.DataFrame, name: str, short: str, long: str, window: int, z_flag: float = 1.0) -> SlopeResult:
    """Compute a slope, its z-score, and DV01-neutral leg sizing.

    yield_panel: date-indexed, columns are tenor labels, values in percent.
    """
    slope_bp = (yield_panel[long] - yield_panel[short]) * 100.0  # % -> bp
    z = _zscore_latest(slope_bp, window)
    as_of = yield_panel.index.max()
    y_short = float(yield_panel[short].iloc[-1])
    y_long = float(yield_panel[long].iloc[-1])
    dv01_s = dv01_par_bond(y_short, TENOR_YEARS[short])
    dv01_l = dv01_par_bond(y_long, TENOR_YEARS[long])

    # z high == historically steep -> mean-reversion favors a flattener.
    if np.isnan(z):
        signal = "insufficient history"
    elif z > z_flag:
        signal = "steep vs history -> flattener (rich)"
    elif z < -z_flag:
        signal = "flat vs history -> steepener (cheap)"
    else:
        signal = "in range"

    return SlopeResult(
        name=name, short=short, long=long,
        slope_bp=float(slope_bp.iloc[-1]), zscore=z, signal=signal,
        dv01_short=dv01_s, dv01_long=dv01_l,
        hedge_ratio_short_per_long=float(dv01_l / dv01_s),
        date=as_of,
    )


@dataclass
class ButterflyResult:
    name: str
    wing_short: str
    belly: str
    wing_long: str
    fly_bp: float
    zscore: float
    signal: str
    dv01_belly: float
    wing_short_units_per_belly: float
    wing_long_units_per_belly: float
    date: pd.Timestamp


def butterfly_screen(yield_panel: pd.DataFrame, name: str, wing_short: str, belly: str, wing_long: str, window: int, z_flag: float = 1.0) -> ButterflyResult:
    """Compute a 2*belly - wings butterfly, z-score, and DV01-neutral wings.

    Wings are sized to split the belly's DV01 50/50, so the fly is a pure
    curvature view (level- and slope-neutral to first order).
    """
    fly_bp = (2.0 * yield_panel[belly] - yield_panel[wing_short] - yield_panel[wing_long]) * 100.0
    z = _zscore_latest(fly_bp, window)
    as_of = yield_panel.index.max()

    y_b = float(yield_panel[belly].iloc[-1])
    y_ws = float(yield_panel[wing_short].iloc[-1])
    y_wl = float(yield_panel[wing_long].iloc[-1])
    dv01_b = dv01_par_bond(y_b, TENOR_YEARS[belly])
    dv01_ws = dv01_par_bond(y_ws, TENOR_YEARS[wing_short])
    dv01_wl = dv01_par_bond(y_wl, TENOR_YEARS[wing_long])

    if np.isnan(z):
        signal = "insufficient history"
    elif z > z_flag:
        signal = "belly cheap vs history -> buy belly / sell wings"
    elif z < -z_flag:
        signal = "belly rich vs history -> sell belly / buy wings"
    else:
        signal = "in range"

    return ButterflyResult(
        name=name, wing_short=wing_short, belly=belly, wing_long=wing_long,
        fly_bp=float(fly_bp.iloc[-1]), zscore=z, signal=signal,
        dv01_belly=dv01_b,
        wing_short_units_per_belly=float(0.5 * dv01_b / dv01_ws),
        wing_long_units_per_belly=float(0.5 * dv01_b / dv01_wl),
        date=as_of,
    )


@dataclass
class CarryRollPoint:
    tenor: str
    yield_pct: float
    horizon_years: float
    roll_yield_bp: float       # yield the bond rolls through over the horizon
    carry_return_pct: float    # running yield earned over the horizon
    rolldown_return_pct: float # price gain from rolling down the curve
    total_return_pct: float
    dv01: float
    total_per_dv01: float


def carry_rolldown_screen(
    latest_yields: Dict[str, float],
    horizon_years: float,
    interpolation: str = "linear",
) -> List[CarryRollPoint]:
    """Carry + rolldown for each tenor over `horizon_years`, ranked per DV01.

    latest_yields: tenor label -> par yield (percent) for the current curve.
    Roll target yield y(T - h) is interpolated from the observed par curve.
    """
    tenors = [t for t in TENOR_ORDER if t in latest_yields and np.isfinite(latest_yields[t])]
    years = np.array([TENOR_YEARS[t] for t in tenors])
    yvals = np.array([latest_yields[t] for t in tenors])

    points: List[CarryRollPoint] = []
    for t in tenors:
        T = TENOR_YEARS[t]
        roll_to = T - horizon_years
        if roll_to <= 0:
            continue  # bond matures within the horizon; roll undefined
        y_T = latest_yields[t]
        y_roll = float(interpolate_par_yields(years, yvals, [roll_to], interpolation)[0])
        moddur = modified_duration_par_bond(y_T, T)
        roll_yield_bp = (y_T - y_roll) * 100.0
        carry_return = y_T * horizon_years
        rolldown_return = moddur * (y_T - y_roll)  # % return from the yield roll
        total = carry_return + rolldown_return
        dv01 = dv01_par_bond(y_T, T)
        points.append(
            CarryRollPoint(
                tenor=t, yield_pct=y_T, horizon_years=horizon_years,
                roll_yield_bp=roll_yield_bp, carry_return_pct=carry_return,
                rolldown_return_pct=rolldown_return, total_return_pct=total,
                dv01=dv01, total_per_dv01=total / dv01 if dv01 else float("nan"),
            )
        )
    points.sort(key=lambda p: p.total_per_dv01, reverse=True)
    return points


def rolldown_for_carry(latest_yields: Dict[str, float], horizon_years: float, interpolation: str = "linear") -> Dict[str, float]:
    """Annualized rolldown return per tenor (percent), for the Phase 2 carry hook.

    rates carry becomes yield + this rolldown, i.e. total running carry+roll.
    """
    points = carry_rolldown_screen(latest_yields, horizon_years, interpolation)
    return {p.tenor: p.rolldown_return_pct / horizon_years for p in points}


def latest_yields_from_panel(yield_panel: pd.DataFrame) -> Dict[str, float]:
    """Extract {tenor: yield} for the most recent date, dropping NaNs."""
    row = yield_panel.iloc[-1]
    return {t: float(row[t]) for t in row.index if pd.notna(row[t])}
