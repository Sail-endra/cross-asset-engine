"""Server-side chart rendering for the briefing, as base64 PNG strings.

Everything is drawn headless (Agg) and embedded inline so the briefing HTML is
a single portable file with no external asset requests -- a hard requirement
from the spec.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def curve_chart_base64(bootstrap_rows: List[Dict]) -> str:
    """Par / zero / implied-forward curves over the bootstrap grid."""
    ts = [r["t"] for r in bootstrap_rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=110)
    ax.plot(ts, [r["par_pct"] for r in bootstrap_rows], label="par", linewidth=1.6)
    ax.plot(ts, [r["zero_pct"] for r in bootstrap_rows], label="zero (spot)", linewidth=1.6)
    ax.plot(ts, [r["fwd_pct"] for r in bootstrap_rows], label="6m forward", linewidth=1.2, alpha=0.8)
    ax.set_title("Treasury curve: par vs bootstrapped zero vs implied forward")
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("yield (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return _fig_to_base64(fig)


def carry_roll_bar_base64(points: List[Dict]) -> str:
    """Bar chart of carry+rolldown per DV01 across tenors (the RV ranking)."""
    tenors = [p["tenor"] for p in points]
    vals = [p["total_per_dv01"] for p in points]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=110)
    ax.bar(range(len(tenors)), vals, color="#3a7bd5")
    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels(tenors, fontsize=8)
    ax.set_title("Carry + rolldown per unit DV01, by tenor")
    ax.set_ylabel("total return % / DV01")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)
