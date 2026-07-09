"""Tests for the briefing narrative and template rendering.

These are pure/offline: they exercise the rules-based narrative and the HTML
render against a synthetic data object, so no network or API key is needed.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from cross_asset_engine.briefing.narrative import build_narrative

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "cross_asset_engine" / "briefing" / "templates"


def _synthetic_data():
    return {
        "as_of": "2026-07-08",
        "generated_at": "2026-07-08T14:00:00+00:00",
        "rv_flag_zscore": 1.0,
        "narrative_source": "rules",
        "regime": {
            "threshold": {"regime": "NORMAL", "risk_asset": "SP500", "vol": 0.14, "z": 0.67},
            "markov": {"regime": "LOW", "p_high": 0.02},
        },
        "levels": [
            {"asset_class": "equities", "asset_id": "SP500", "value": 7504.0, "change": -33.6, "source": "fred"},
            {"asset_class": "credit", "asset_id": "US_HY_OAS", "value": 2.72, "change": -0.02, "source": "fred"},
            {"asset_class": "rates", "asset_id": "3M", "value": 3.87, "change": 0.05, "source": "fred"},
        ],
        "signal_stats": {
            "Equity momentum (126d)": {"family": "momentum", "hit": 0.565, "sharpe_net": 0.19,
                                       "ann_ret_net": 0.039, "maxdd": -0.66, "n": 437},
            "FX carry": {"family": "carry", "hit": 0.50, "sharpe_net": -0.08,
                         "ann_ret_net": -0.005, "maxdd": -0.29, "n": 238},
        },
        "curve": {
            "bootstrap": [],
            "slopes": [{"name": "2s10s", "short": "2Y", "long": "10Y", "slope_bp": 35.0,
                        "zscore": -2.11, "signal": "flat vs history -> steepener (cheap)",
                        "hedge_ratio_short_per_long": 4.2, "dv01_short": 0.019, "dv01_long": 0.08}],
            "butterflies": [],
            "carry_roll": [{"tenor": "1Y", "yield_pct": 3.95, "roll_yield_bp": -1.5,
                            "carry_return_pct": 0.98, "rolldown_return_pct": -0.02,
                            "total_return_pct": 0.96, "dv01": 0.01, "total_per_dv01": 100.0}],
            "richest": "6M", "cheapest": "30Y", "horizon_months": 3,
        },
    }


def test_narrative_is_numbers_driven_and_calibrated():
    text = build_narrative(_synthetic_data())
    # tone + largest mover
    assert "risk-on" in text
    assert "SP500" in text and "14.0%" in text
    # conviction calibration: positive-but-small Sharpe -> low-conviction,
    # negative Sharpe -> flagged watch item
    assert "low-conviction" in text
    assert "flagged watch item" in text
    # DV01-neutral sizing surfaced for the flagged steepener
    assert "DV01-neutral" in text and "4.20" in text
    # front-end inversion caveat when rolldown is negative
    assert "negative rolldown" in text


def test_narrative_flags_equity_credit_divergence():
    # equities down but HY tighter -> divergence should be called out
    text = build_narrative(_synthetic_data())
    assert "diverging" in text


def test_template_renders_without_external_assets():
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    data = _synthetic_data()
    data["signals_today"] = {"momentum": [], "carry": []}
    data["narrative"] = build_narrative(data)
    html = env.get_template("briefing.html.j2").render(data=data, chart_curve="AAAA", chart_roll="BBBB")
    assert "<!doctype html>" in html.lower()
    # no external resource references (only inline data: URIs allowed)
    assert "http://" not in html and "https://" not in html
    assert "data:image/png;base64,AAAA" in html
