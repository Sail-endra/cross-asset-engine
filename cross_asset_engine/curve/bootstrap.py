"""Bootstrap a zero-coupon (spot) curve from Treasury par yields.

Finance rationale
-----------------
FRED publishes constant-maturity Treasury (CMT) *par* yields -- the coupon a
bond would need to trade at par for each maturity. To discount arbitrary cash
flows (and to compute forwards, rolldown, and DV01 properly) we need the
*zero-coupon* curve: the rate for a single payment at each horizon. Bootstrapping
recovers it one maturity at a time from the short end out, because each par
bond's price depends on the discount factors of its earlier coupons, which are
already solved by the time we reach it.

Method
------
1. Interpolate the observed par curve onto a semiannual grid (Treasuries pay
   semiannually), linear-on-yield by default, cubic optionally.
2. Bootstrap discount factors on that grid. The first node (0.5y) is a single
   payment, hence zero-coupon. Each later node solves the par condition
       1 = (y_n/2) * sum_{i<n} DF_i  +  (1 + y_n/2) * DF_n
   for DF_n, given the already-solved earlier DFs.
3. Convert DFs to semiannually-compounded zero rates and derive implied
   forwards from ratios of discount factors.

A flat par curve must bootstrap to an identically flat zero curve (proved in
the tests via a hand-computed example), which is the cleanest correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_par_yields(
    tenors_years: Sequence[float],
    yields_pct: Sequence[float],
    grid: Sequence[float],
    method: str = "linear",
) -> np.ndarray:
    """Interpolate observed par yields onto a target maturity grid.

    linear -> piecewise-linear on yield (the market-standard default for CMT);
    cubic  -> natural cubic spline (smoother forwards, can overshoot).
    Flat extrapolation past the observed endpoints (never wild extrapolation).
    """
    t = np.asarray(tenors_years, dtype=float)
    y = np.asarray(yields_pct, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    g = np.asarray(grid, dtype=float)
    if method == "linear":
        return np.interp(g, t, y)  # np.interp clamps (flat) outside the range
    if method == "cubic":
        cs = CubicSpline(t, y, bc_type="natural", extrapolate=True)
        out = cs(g)
        # clamp extrapolation to endpoints to avoid spline blow-ups past the ends
        out = np.where(g < t[0], y[0], out)
        out = np.where(g > t[-1], y[-1], out)
        return out
    raise ValueError(f"Unknown interpolation method: {method!r}")


@dataclass
class ZeroCurve:
    """A bootstrapped zero curve: grid times, discount factors, zero rates.

    Discount factors are interpolated log-linearly (piecewise-constant forward
    rate), the standard well-behaved choice for querying between nodes.
    """

    times: np.ndarray          # semiannual grid, years
    discount_factors: np.ndarray
    zero_rates_pct: np.ndarray  # semiannually compounded, percent
    par_yields_pct: np.ndarray  # the (interpolated) par curve it was built from

    def df(self, t: float) -> float:
        """Discount factor at time t (years), log-linear interpolation on DF."""
        if t <= 0:
            return 1.0
        times = np.concatenate([[0.0], self.times])
        logdf = np.concatenate([[0.0], np.log(self.discount_factors)])
        if t >= times[-1]:
            # extend at the last node's forward (flat continuously-comp forward)
            f = -(logdf[-1] - logdf[-2]) / (times[-1] - times[-2])
            return float(np.exp(logdf[-1] - f * (t - times[-1])))
        return float(np.exp(np.interp(t, times, logdf)))

    def zero_rate(self, t: float) -> float:
        """Semiannually-compounded zero rate at t (percent)."""
        if t <= 0:
            return float(self.zero_rates_pct[0])
        d = self.df(t)
        return float(2.0 * (d ** (-1.0 / (2.0 * t)) - 1.0) * 100.0)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Semiannually-compounded forward rate over [t1, t2] (percent)."""
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        ratio = self.df(t1) / self.df(t2)
        return float(2.0 * (ratio ** (1.0 / (2.0 * (t2 - t1))) - 1.0) * 100.0)


def bootstrap_zero_curve(
    tenors_years: Sequence[float],
    par_yields_pct: Sequence[float],
    step_years: float = 0.5,
    interpolation: str = "linear",
    max_maturity: float | None = None,
) -> ZeroCurve:
    """Bootstrap a ZeroCurve from par yields.

    See module docstring for the method. Assumes semiannual coupons (step 0.5).
    """
    tenors_years = np.asarray(tenors_years, dtype=float)
    par_yields_pct = np.asarray(par_yields_pct, dtype=float)
    finite = np.isfinite(tenors_years) & np.isfinite(par_yields_pct)
    tenors_years, par_yields_pct = tenors_years[finite], par_yields_pct[finite]
    if len(tenors_years) < 2:
        raise ValueError("Need at least two par-yield points to bootstrap")

    top = max_maturity or float(tenors_years.max())
    n_nodes = int(round(top / step_years))
    grid = np.array([step_years * (i + 1) for i in range(n_nodes)])

    y = interpolate_par_yields(tenors_years, par_yields_pct, grid, interpolation) / 100.0

    dfs = np.empty(n_nodes)
    for i in range(n_nodes):
        c = y[i] / 2.0  # semiannual coupon as a fraction of face (par bond)
        if i == 0:
            dfs[i] = 1.0 / (1.0 + c)
        else:
            dfs[i] = (1.0 - c * dfs[:i].sum()) / (1.0 + c)

    zero_rates = 2.0 * (dfs ** (-1.0 / (2.0 * grid)) - 1.0) * 100.0
    return ZeroCurve(
        times=grid,
        discount_factors=dfs,
        zero_rates_pct=zero_rates,
        par_yields_pct=y * 100.0,
    )


def implied_forward_curve(curve: ZeroCurve) -> np.ndarray:
    """Period-by-period implied forward rates between consecutive grid nodes (%)."""
    forwards = np.empty(len(curve.times))
    forwards[0] = curve.zero_rates_pct[0]
    for i in range(1, len(curve.times)):
        forwards[i] = curve.forward_rate(curve.times[i - 1], curve.times[i])
    return forwards
