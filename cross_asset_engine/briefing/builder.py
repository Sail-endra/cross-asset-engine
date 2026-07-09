"""Assemble the daily briefing: one data object -> HTML page + JSON snapshot.

The data object is the single source of truth. It is serialized verbatim to a
dated JSON file, and the HTML is rendered from the same object, so every number
shown in the briefing is traceable to a field in the JSON (the auditability
requirement). Charts are embedded inline as base64 PNGs, so the HTML file is
fully portable with no external asset dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..data.market_data import MarketData
from ..settings import load_params
from ..signals.base import pivot_to_panel
from ..signals import SIGNAL_REGISTRY
from ..signals import regime as regime_mod
from ..tenors import TENOR_YEARS, TENOR_ORDER
from ..curve.bootstrap import bootstrap_zero_curve, implied_forward_curve
from ..curve.rv_screens import (
    slope_screen, butterfly_screen, carry_rolldown_screen, latest_yields_from_panel,
)
from ..backtest.engine import run_backtest
from ..backtest.strategies import build_default_backtests_inputs
from . import charts, narrative as narrative_mod

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _signal_family_key(name: str) -> str:
    n = name.lower()
    return "momentum" if "momentum" in n else "carry" if "carry" in n else "curve"


def assemble_data(md: MarketData, params: Dict) -> Dict:
    """Compute every number the briefing shows and return it as one dict."""
    bt = params["backtest"]
    cparams = params["curve"]

    # --- levels & moves -----------------------------------------------------
    summary = md.latest_levels_and_changes()
    levels = summary.to_dict(orient="records")
    for r in levels:
        r["date"] = str(pd.Timestamp(r["date"]).date())
        r["value"] = float(r["value"])
        r["change"] = float(r["change"])

    # --- regime -------------------------------------------------------------
    thr = regime_mod.current_regime(md, params, method="threshold")
    mk = regime_mod.current_regime(md, params, method="markov")
    regime = {
        "threshold": {"regime": thr["regime"], "risk_asset": thr["risk_asset"],
                      "vol": thr["vol"], "z": thr["z"], "date": str(thr["date"].date())},
        "markov": {"regime": mk["regime"], "p_high": mk["p_high"], "date": str(mk["date"].date())},
    }

    # --- signals today (momentum ts_126d + carry) ---------------------------
    mom = [s.to_dict() for s in SIGNAL_REGISTRY["momentum"](md, params) if s.variant == "ts_126d"]
    carry = [s.to_dict() for s in SIGNAL_REGISTRY["carry"](md, params)]
    signals_today = {"momentum": mom, "carry": carry}

    # --- backtested stats per family ----------------------------------------
    signal_stats: Dict[str, Dict] = {}
    for name, (pos, px, units) in build_default_backtests_inputs(md, params).items():
        res = run_backtest(name, pos, px, bt["horizon_days"], bt["cost_bps"], bt["risk_free_pct"], units)
        g, n = res.metrics_gross, res.metrics_net
        signal_stats[name] = {
            "family": _signal_family_key(name),
            "hit": g.hit_rate, "sharpe_gross": g.sharpe, "sharpe_net": n.sharpe,
            "ann_ret_net": n.ann_return, "maxdd": g.max_drawdown, "n": g.n_periods,
        }

    # --- curve: bootstrap + RV screens --------------------------------------
    panel = pivot_to_panel(md.get_curve())
    latest = latest_yields_from_panel(panel)
    tenors = [t for t in TENOR_ORDER if t in latest]
    years = [TENOR_YEARS[t] for t in tenors]
    yields = [latest[t] for t in tenors]
    zc = bootstrap_zero_curve(years, yields, cparams["bootstrap_step_years"],
                              cparams["interpolation"], max_maturity=max(years))
    fwd = implied_forward_curve(zc)
    bootstrap_rows = [
        {"t": float(zc.times[i]), "par_pct": float(zc.par_yields_pct[i]),
         "zero_pct": float(zc.zero_rates_pct[i]), "fwd_pct": float(fwd[i]),
         "df": float(zc.discount_factors[i])}
        for i in range(len(zc.times))
    ]

    window = cparams["zscore_window"]
    slopes = []
    for s in cparams["slopes"]:
        if s["short"] in panel and s["long"] in panel:
            r = slope_screen(panel, s["name"], s["short"], s["long"], window, params["briefing"]["rv_flag_zscore"])
            slopes.append({"name": r.name, "short": r.short, "long": r.long, "slope_bp": r.slope_bp,
                           "zscore": r.zscore, "signal": r.signal,
                           "hedge_ratio_short_per_long": r.hedge_ratio_short_per_long,
                           "dv01_short": r.dv01_short, "dv01_long": r.dv01_long})
    flies = []
    for b in cparams["butterflies"]:
        r = butterfly_screen(panel, b["name"], b["wing_short"], b["belly"], b["wing_long"], window,
                             params["briefing"]["rv_flag_zscore"])
        flies.append({"name": r.name, "fly_bp": r.fly_bp, "zscore": r.zscore, "signal": r.signal})

    horizon = cparams["carry_rolldown_horizon_months"] / 12.0
    points = carry_rolldown_screen(latest, horizon, cparams["interpolation"])
    carry_roll = [
        {"tenor": p.tenor, "yield_pct": p.yield_pct, "roll_yield_bp": p.roll_yield_bp,
         "carry_return_pct": p.carry_return_pct, "rolldown_return_pct": p.rolldown_return_pct,
         "total_return_pct": p.total_return_pct, "dv01": p.dv01, "total_per_dv01": p.total_per_dv01}
        for p in points
    ]

    curve = {
        "bootstrap": bootstrap_rows,
        "slopes": slopes,
        "butterflies": flies,
        "carry_roll": carry_roll,
        "richest": points[0].tenor if points else None,
        "cheapest": points[-1].tenor if points else None,
        "horizon_months": cparams["carry_rolldown_horizon_months"],
    }

    as_of = str(md.as_of)
    return {
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rv_flag_zscore": params["briefing"]["rv_flag_zscore"],
        "regime": regime,
        "levels": levels,
        "signals_today": signals_today,
        "signal_stats": signal_stats,
        "curve": curve,
    }


def build_briefing(project_root: Path, use_llm: bool = False, output_dir: Path | None = None) -> Dict[str, Path]:
    """Run the full pipeline and write index.html + a dated JSON snapshot."""
    md = MarketData(project_root=project_root)
    params = load_params()
    data = assemble_data(md, params)

    # narrative (rules-based default; LLM only when explicitly requested)
    if use_llm:
        from .llm_narrative import llm_narrative
        data["narrative"] = llm_narrative(data, params["briefing"]["llm_model"])
        data["narrative_source"] = "llm"
    else:
        data["narrative"] = narrative_mod.build_narrative(data)
        data["narrative_source"] = "rules"

    # charts embedded inline
    chart_curve = charts.curve_chart_base64(data["curve"]["bootstrap"])
    chart_roll = charts.carry_roll_bar_base64(data["curve"]["carry_roll"])

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("briefing.html.j2")
    html = template.render(data=data, chart_curve=chart_curve, chart_roll=chart_roll)

    out = output_dir or (project_root / "docs")
    out.mkdir(parents=True, exist_ok=True)
    date = data["as_of"]
    (out / "index.html").write_text(html)
    (out / f"briefing_{date}.html").write_text(html)
    json_path = out / f"briefing_{date}.json"
    json_path.write_text(json.dumps(data, indent=2, default=str))
    return {"html": out / "index.html", "json": json_path}
