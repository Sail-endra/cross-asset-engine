"""DV01: the dollar value of a one-basis-point yield move.

Finance rationale
-----------------
DV01 (a.k.a. PV01 / the "dollar duration" of a basis point) is the sizing
primitive for every relative-value trade in this project. A 2s10s steepener or
a 2s5s10s butterfly is only a pure view on *slope* or *curvature* if each leg
carries the same interest-rate risk -- otherwise it's contaminated by a
directional (level) bet. DV01 is how we equalize that risk: size legs so their
DV01s offset. It is computed here by the honest, model-free route -- reprice
the instrument with its yield bumped one basis point and take the difference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bootstrap import ZeroCurve

ONE_BP = 1e-4


def par_bond_price(coupon_pct: float, ytm_pct: float, tenor_years: float, freq: int = 2, face: float = 100.0) -> float:
    """Price a coupon bond off a single yield-to-maturity (semiannual by default)."""
    c = coupon_pct / 100.0
    y = ytm_pct / 100.0
    n = int(round(tenor_years * freq))
    if n <= 0:
        return face
    coupon = face * c / freq
    times = np.arange(1, n + 1)
    discount = (1.0 + y / freq) ** (-times)
    return float(coupon * discount.sum() + face * discount[-1])


def modified_duration_par_bond(ytm_pct: float, tenor_years: float, freq: int = 2, face: float = 100.0) -> float:
    """Modified duration of a par bond (coupon == ytm), in years."""
    y = ytm_pct / 100.0
    n = int(round(tenor_years * freq))
    if n <= 0:
        return 0.0
    coupon = face * (y / freq)
    times = np.arange(1, n + 1)
    cfs = np.full(n, coupon)
    cfs[-1] += face
    discount = (1.0 + y / freq) ** (-times)
    pv = cfs * discount
    price = pv.sum()
    macaulay = (times / freq * pv).sum() / price
    return macaulay / (1.0 + y / freq)


def dv01_par_bond(ytm_pct: float, tenor_years: float, freq: int = 2, face: float = 100.0) -> float:
    """DV01 of a par bond at `ytm_pct`, by a symmetric 1bp yield bump.

    Coupon is set equal to the yield (the bond is at par), then the yield is
    bumped +/-1bp and the price change measured. Returns a positive number:
    dollars of price change per 1bp, per `face` of notional.
    """
    coupon = ytm_pct  # par bond
    p_up = par_bond_price(coupon, ytm_pct + ONE_BP * 100, tenor_years, freq, face)
    p_dn = par_bond_price(coupon, ytm_pct - ONE_BP * 100, tenor_years, freq, face)
    return float((p_dn - p_up) / 2.0)


def dv01_from_cashflows(times_years: Sequence[float], cashflows: Sequence[float], curve: ZeroCurve) -> float:
    """DV01 of an arbitrary cashflow stream priced off the zero curve.

    Reprices with every zero rate shifted +/-1bp (parallel) and returns the
    per-1bp price change. Model-free given the curve; used for non-par positions.
    """
    times_years = np.asarray(times_years, dtype=float)
    cashflows = np.asarray(cashflows, dtype=float)

    def price_with_shift(shift_bp: float) -> float:
        total = 0.0
        for t, cf in zip(times_years, cashflows):
            z = (curve.zero_rate(t) + shift_bp) / 100.0
            df = (1.0 + z / 2.0) ** (-2.0 * t)
            total += cf * df
        return total

    p_up = price_with_shift(+1.0 * ONE_BP * 100)
    p_dn = price_with_shift(-1.0 * ONE_BP * 100)
    return float((p_dn - p_up) / 2.0)
