"""Market-regime detection, used to gate/scale the other signals.

Finance rationale
-----------------
Carry and trend break at different times. Carry -- being paid to hold risk --
does well in calm markets and gets hurt precisely when volatility spikes and
everyone reaches for the same exit (risk-off). A regime read lets us scale
carry exposure down when the environment turns stressed, which is the single
most important risk control for a carry book. This module produces that read
two ways and exposes a `carry_scale` multiplier the carry signal applies.

Two methods (ported from the Perihelion volatility project's
regimeDetection.js, kept deliberately faithful so the two projects tell one
story):

A. Threshold -- z-score EWMA realized volatility against its own trailing
   1-year window and bucket into LOW / NORMAL / HIGH / CRISIS.
B. Markov -- fit a 2-state Gaussian hidden Markov model to returns (a calm,
   low-variance state and a turbulent, high-variance state) and map the
   smoothed probability of the high-variance state onto the same buckets.

Both are strictly causal in how they're consumed: `current_regime` reads the
last row, and the per-date series can be lagged by the backtester so a day's
regime is only acted on the following day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REGIMES = ["LOW", "NORMAL", "HIGH", "CRISIS"]
_ROLLING_WINDOW = 252
_MIN_WINDOW = 30


# ---- Threshold buckets (identical thresholds to the Perihelion project) ----

def regime_from_z(z: float) -> str:
    if z < -0.5:
        return "LOW"
    if z < 0.75:
        return "NORMAL"
    if z < 1.5:
        return "HIGH"
    return "CRISIS"


def regime_from_p_high(p_high: float) -> str:
    if p_high < 0.2:
        return "LOW"
    if p_high < 0.5:
        return "NORMAL"
    if p_high < 0.8:
        return "HIGH"
    return "CRISIS"


def ewma_realized_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Annualized RiskMetrics-style EWMA volatility of a daily return series.

    var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2  (uses only past returns), then
    annualized by sqrt(252). Recursive and causal.
    """
    r = returns.dropna()
    var = np.empty(len(r))
    if len(r) == 0:
        return pd.Series(dtype=float)
    var[0] = r.iloc[0] ** 2
    r_vals = r.to_numpy()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r_vals[t - 1] ** 2
    return pd.Series(np.sqrt(var * 252.0), index=r.index)


def classify_threshold(vol: pd.Series) -> pd.DataFrame:
    """Method A: z-score EWMA vol vs a trailing 252-day window, then bucket.

    Expanding window until a full year of history exists (min 30 obs), matching
    the Perihelion implementation. Returns columns [vol, z, regime].
    """
    out = []
    vol = vol.dropna()
    vals = vol.to_numpy()
    for i in range(len(vals)):
        start = max(0, i - _ROLLING_WINDOW + 1)
        window = vals[start : i + 1]
        if len(window) < _MIN_WINDOW:
            out.append((vol.index[i], vals[i], 0.0, "NORMAL"))
            continue
        mean = window.mean()
        sd = window.std(ddof=1)
        z = (vals[i] - mean) / sd if sd > 0 else 0.0
        out.append((vol.index[i], vals[i], z, regime_from_z(z)))
    return pd.DataFrame(out, columns=["date", "vol", "z", "regime"]).set_index("date")


# ---- Method B: 2-state Gaussian HMM (self-contained, deterministic) --------

@dataclass
class MarkovRegimeModel:
    """Two-state Gaussian HMM fit by Baum-Welch EM on a return series.

    State 0 is the calm/low-variance regime, state 1 the turbulent/high-
    variance regime (enforced by ordering the fitted variances). Initialization
    is deterministic -- variances seeded at 0.5x and 2x the sample variance,
    a sticky transition matrix, equal priors -- so no RNG seed is needed and
    results are identical run to run, per the reproducibility constraint.
    """

    n_iter: int = 100
    tol: float = 1e-6
    mu: np.ndarray = None
    var: np.ndarray = None
    trans: np.ndarray = None
    pi: np.ndarray = None

    def _emission(self, x: np.ndarray) -> np.ndarray:
        # Gaussian pdf per state; small floor on variance for stability.
        v = np.maximum(self.var, 1e-12)
        coef = 1.0 / np.sqrt(2.0 * np.pi * v)
        return coef * np.exp(-((x[:, None] - self.mu[None, :]) ** 2) / (2.0 * v[None, :]))

    def fit(self, returns: pd.Series) -> "MarkovRegimeModel":
        x = returns.dropna().to_numpy(dtype=float)
        n = len(x)
        if n < _MIN_WINDOW:
            raise ValueError("Not enough observations to fit the Markov regime model")
        sample_var = x.var(ddof=1)
        self.mu = np.array([x.mean(), x.mean()])
        self.var = np.array([0.5 * sample_var, 2.0 * sample_var])
        self.trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.pi = np.array([0.5, 0.5])

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = self._emission(x)  # (n, 2)
            # Forward-backward with scaling
            alpha = np.zeros((n, 2))
            c = np.zeros(n)
            alpha[0] = self.pi * B[0]
            c[0] = alpha[0].sum()
            alpha[0] /= c[0] + 1e-300
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ self.trans) * B[t]
                c[t] = alpha[t].sum()
                alpha[t] /= c[t] + 1e-300
            beta = np.zeros((n, 2))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = (self.trans @ (B[t + 1] * beta[t + 1])) / (c[t + 1] + 1e-300)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi_sum = np.zeros((2, 2))
            for t in range(n - 1):
                denom = (alpha[t] @ self.trans) * B[t + 1] @ beta[t + 1]
                for i in range(2):
                    num = alpha[t, i] * self.trans[i] * B[t + 1] * beta[t + 1]
                    xi_sum[i] += num / (denom + 1e-300)

            self.pi = gamma[0]
            self.trans = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)
            w = gamma / (gamma.sum(axis=0, keepdims=True) + 1e-300)
            self.mu = (w * x[:, None]).sum(axis=0)
            self.var = (w * (x[:, None] - self.mu[None, :]) ** 2).sum(axis=0)

            ll = np.log(c + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Order states so index 1 is always the high-variance ("risk-off") one.
        if self.var[0] > self.var[1]:
            self._swap_states()
        self._gamma = gamma
        self._index = returns.dropna().index
        return self

    def _swap_states(self) -> None:
        self.mu = self.mu[::-1].copy()
        self.var = self.var[::-1].copy()
        self.pi = self.pi[::-1].copy()
        self.trans = self.trans[::-1, ::-1].copy()

    def prob_high_vol(self) -> pd.Series:
        """Smoothed P(high-variance state) per date, after state ordering."""
        p = self._gamma.copy()
        if self.var[0] > self.var[1]:  # already ordered in fit, guard anyway
            p = p[:, ::-1]
        # After ordering, high-vol is column 1.
        high = p[:, 1] if self.var[1] >= self.var[0] else p[:, 0]
        return pd.Series(high, index=self._index)


def classify_markov(returns: pd.Series) -> pd.DataFrame:
    """Method B convenience wrapper: fit and bucket P(high-vol) into regimes."""
    model = MarkovRegimeModel().fit(returns)
    p_high = model.prob_high_vol()
    regimes = p_high.apply(regime_from_p_high)
    return pd.DataFrame({"p_high": p_high, "regime": regimes})


# ---- Public API used by the rest of the engine ----------------------------

def _returns_from_levels(levels: pd.Series) -> pd.Series:
    return levels.sort_index().pct_change().dropna()


def current_regime(
    market_data, params: Dict, method: str = "threshold", as_of: Optional[pd.Timestamp] = None
) -> Dict:
    """Latest regime label (and its driver) for the configured risk asset."""
    risk_asset = params["signals"]["regime"]["risk_asset"]
    lam = params["signals"]["regime"]["ewma_lambda"]
    eq = market_data.get_equities()
    levels = eq[eq["asset_id"] == risk_asset].set_index("date")["value"].sort_index()
    if as_of is not None:
        levels = levels.loc[:as_of]
    returns = _returns_from_levels(levels)

    if method == "threshold":
        vol = ewma_realized_vol(returns, lam)
        classified = classify_threshold(vol)
        last = classified.iloc[-1]
        return {
            "method": "threshold",
            "risk_asset": risk_asset,
            "date": classified.index[-1],
            "regime": last["regime"],
            "vol": float(last["vol"]),
            "z": float(last["z"]),
        }
    elif method == "markov":
        classified = classify_markov(returns)
        last = classified.iloc[-1]
        return {
            "method": "markov",
            "risk_asset": risk_asset,
            "date": classified.index[-1],
            "regime": last["regime"],
            "p_high": float(last["p_high"]),
        }
    raise ValueError(f"Unknown regime method: {method!r}")


def carry_scale(regime: str, params: Dict) -> float:
    """Multiplier in [0,1] applied to carry exposure given the regime.

    Scales carry down as the regime worsens -- the core risk control from the
    module docstring. Values are config-driven (params.signals.regime.carry_scale).
    """
    scales = params["signals"]["regime"]["carry_scale"]
    if regime not in scales:
        raise ValueError(f"No carry_scale configured for regime {regime!r}")
    return float(scales[regime])


def regime_series(
    market_data, params: Dict, method: str = "threshold", as_of: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Full per-date regime history (for charts / backtests)."""
    risk_asset = params["signals"]["regime"]["risk_asset"]
    lam = params["signals"]["regime"]["ewma_lambda"]
    eq = market_data.get_equities()
    levels = eq[eq["asset_id"] == risk_asset].set_index("date")["value"].sort_index()
    if as_of is not None:
        levels = levels.loc[:as_of]
    returns = _returns_from_levels(levels)
    if method == "threshold":
        return classify_threshold(ewma_realized_vol(returns, lam))
    if method == "markov":
        return classify_markov(returns)
    raise ValueError(f"Unknown regime method: {method!r}")
