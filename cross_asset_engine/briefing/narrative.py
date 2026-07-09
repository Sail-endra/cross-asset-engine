"""Rules-based narrative generation -- the default briefing prose.

Turns the computed numbers into a readable sell-side-style morning note without
any model call. Every sentence is derived mechanically from a field in the data
object, so it can never invent a figure or assert a cause for a move (there is
no news feed). The optional LLM path (llm_narrative.py) is held to the same
constraints via its system prompt.
"""

from __future__ import annotations

from typing import Dict, List


def _pct_change(value: float, change: float) -> float | None:
    prior = value - change
    if prior == 0:
        return None
    return change / prior * 100.0


def _conviction(sharpe: float | None) -> str:
    if sharpe is None:
        return "no backtest available"
    if sharpe > 0.2:
        return "actionable"
    if sharpe >= -0.05:
        return "low-conviction"
    return "flagged watch item (historically unprofitable in-sample -- caveat stands)"


def build_narrative(data: Dict) -> str:
    blocks: List[str] = []
    reg = data["regime"]
    levels = data["levels"]
    stats = data["signal_stats"]
    curve = data["curve"]

    # (1) top-line risk tone + largest mover -----------------------------------
    thr, mk = reg["threshold"], reg["markov"]
    tone = "risk-on / calm" if thr["regime"] in ("LOW", "NORMAL") else "risk-off / stressed"
    # largest equity mover by percent
    eq = [l for l in levels if l["asset_class"] == "equities"]
    movers = []
    for l in eq:
        pc = _pct_change(l["value"], l["change"])
        if pc is not None:
            movers.append((abs(pc), l["asset_id"], pc))
    largest = max(movers) if movers else None
    tone_line = (
        f"Cross-asset tone reads {tone}: realized vol on {thr['risk_asset']} at "
        f"{thr['vol'] * 100:.1f}% (z={thr['z']:.2f}, {thr['regime']}), Markov "
        f"P(high-vol)={mk['p_high']:.2f}."
    )
    if largest:
        tone_line += f" Largest equity move: {largest[1]} {largest[2]:+.2f}%."
    blocks.append(tone_line)

    # (2) one line per asset that moved meaningfully ---------------------------
    lines = []
    for l in levels:
        ac = l["asset_class"]
        if ac in ("equities", "fx"):
            pc = _pct_change(l["value"], l["change"])
            if pc is not None and abs(pc) >= 0.3:
                lines.append(f"{l['asset_id']}: {pc:+.2f}% to {l['value']:.4g}.")
        else:  # rates / credit, moves quoted in bp (values already in %)
            bp = l["change"] * 100.0
            if abs(bp) >= 2.0:
                verb = "cheapened" if bp > 0 else "richened"
                lines.append(f"{l['asset_id']}: {verb} {bp:+.1f}bp to {l['value']:.2f}%.")
    blocks.append(" ".join(lines) if lines else "No single asset moved beyond routine daily ranges.")

    # (3) cross-asset synthesis ------------------------------------------------
    eq_up = sum(1 for l in eq if l["change"] > 0)
    eq_dn = sum(1 for l in eq if l["change"] < 0)
    hy = next((l for l in levels if l["asset_id"] == "US_HY_OAS"), None)
    hy_dir = "wider" if hy and hy["change"] > 0 else "tighter" if hy else "flat"
    risk_on_equities = eq_up >= eq_dn
    consistent = (risk_on_equities and hy_dir == "tighter") or (not risk_on_equities and hy_dir == "wider")
    synth = (
        f"Equities {'mostly higher' if risk_on_equities else 'mostly lower'} with HY OAS {hy_dir}: "
    )
    if consistent:
        synth += "the moves line up as a coherent risk configuration."
    else:
        synth += ("equities and credit are diverging -- one is not confirming the other, a divergence "
                  "worth flagging rather than fading blindly.")
    blocks.append(synth)

    # (4) active signals as trade ideas ----------------------------------------
    ideas = []
    for name, s in stats.items():
        sharpe = s.get("sharpe_net")
        hit = s.get("hit")
        conv = _conviction(sharpe)
        risk = {
            "momentum": "reversal / whipsaw at turning points",
            "carry": "gap risk in a vol spike (regime filter scales this down)",
            "curve": "curve trending rather than mean-reverting",
        }
        key = "momentum" if "momentum" in name.lower() else "carry" if "carry" in name.lower() else "curve"
        ideas.append(
            f"{name}: {conv}; backtested hit {hit * 100:.0f}%, net Sharpe {sharpe:+.2f}. "
            f"Primary risk: {risk[key]}."
        )
    blocks.append(" ".join(ideas))

    # (5) flagged curve RV trade ----------------------------------------------
    rv = []
    for sl in curve["slopes"]:
        if abs(sl["zscore"]) >= data["rv_flag_zscore"]:
            rv.append(
                f"{sl['name']} at {sl['slope_bp']:.0f}bp (z={sl['zscore']:.2f}) -- {sl['signal']}; "
                f"express DV01-neutral with {sl['hedge_ratio_short_per_long']:.2f} units of the front leg "
                f"per unit of the back leg."
            )
    cr = curve["carry_roll"]
    if cr:
        rv.append(
            f"Carry-and-rolldown ranks {curve['richest']} richest and {curve['cheapest']} cheapest per DV01 "
            f"over a {curve['horizon_months']}m horizon."
        )
        neg_roll = [p["tenor"] for p in cr if p["roll_yield_bp"] < 0]
        if neg_roll:
            rv.append(
                f"Caveat: the front of the curve is inverted, so {', '.join(neg_roll)} roll(s) *up* the "
                f"curve -- negative rolldown, not a data error."
            )
    blocks.append(" ".join(rv) if rv else "No curve RV screen is beyond its flag threshold today.")

    return "\n\n".join(blocks)
