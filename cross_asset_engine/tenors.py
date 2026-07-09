"""Canonical mapping of curve tenor labels to maturities in years.

Single source of truth shared by the momentum TR proxy and the curve module,
so a tenor's year-fraction is never defined in two places that could drift.
"""

from __future__ import annotations

TENOR_YEARS = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}

# Ascending maturity order, for iterating the curve short-to-long.
TENOR_ORDER = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
