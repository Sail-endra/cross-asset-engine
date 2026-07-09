"""Optional LLM narrative pass (opt-in, behind the --llm flag).

Design constraints, enforced here and in the system prompt:
  * The model receives ONLY the computed numbers, serialized as JSON, in the
    user message. It has no tools and no network -- it cannot fetch anything.
  * The system prompt forbids inventing figures or asserting causes for moves.
  * anthropic is imported lazily so the default (rules-based) path never
    requires the package or an API key.

The system prompt below is the exact desk-note specification supplied for this
project; keep it verbatim so the narrative behaviour is auditable.
"""

from __future__ import annotations

import json
import os
from typing import Dict

SYSTEM_PROMPT = """\
You are writing a daily cross-asset market briefing in the style of a sell-side
markets desk morning note. You will be given a structured data object containing
today's levels and moves across rates, FX, equities, and credit; the day's signal
outputs (momentum, carry, regime); their backtested statistics (hit rate, Sharpe,
max drawdown); and the yield-curve relative-value screen (slopes, butterfly,
carry-and-rolldown, each with z-scores and DV01-neutral sizing).

ABSOLUTE RULES:
- Use ONLY numbers present in the provided data object. Never invent, estimate,
  or round beyond what is given. Every quantitative claim must be traceable to a
  field in the data.
- You have NO news feed. Never state or imply a cause for any move (no central
  banks, no data releases, no headlines). You may describe what the cross-asset
  pattern is CONSISTENT WITH (e.g. "a risk-off configuration"), framed as an
  observation, never as a confirmed cause.
- Calibrate conviction to the backtested statistics you are given. Present a
  signal with a positive Sharpe as actionable; a flat one as low-conviction; a
  historically unprofitable one as a flagged watch item with the caveat stated
  explicitly. Do not overclaim on a weak signal.
- If a field is absent, omit that topic. Do not fill gaps.

STYLE:
- Terse, quantified, active voice. Short sentences. Correct desk vocabulary
  (bps, richen/cheapen, steepener/flattener, carry, rolldown, DV01-neutral,
  risk-on/off). No filler, no hype, no exclamation, no invented context.
- 180 to 300 words.

STRUCTURE: (1) top-line risk tone and the largest mover; (2) one line per asset
that moved meaningfully; (3) a cross-asset synthesis paragraph stating whether the
assets are telling a consistent story or diverging, and flagging any divergence;
(4) active signals as trade ideas, each with a one-line thesis, direction,
backtested hit rate and Sharpe, and its primary risk; (5) the flagged curve RV
trade with z-score and DV01-neutral sizing, plus any caveat the data surfaces.

Output plain prose in these five short blocks. No preamble, no sign-off.
"""


def llm_narrative(data: Dict, model: str = "claude-opus-4-8") -> str:
    """Generate the narrative from `data` via the Anthropic API.

    Raises a clear error if the key or package is missing -- never silently
    falls back, so an operator who asked for the LLM pass knows it didn't run.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; the --llm narrative path requires it. "
            "Run without --llm for the rules-based narrative."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("The 'anthropic' package is required for the --llm path.") from exc

    client = anthropic.Anthropic()
    # Numbers passed explicitly in the user turn -- the model fetches nothing.
    payload = json.dumps(data, indent=2, default=str)
    response = client.messages.create(
        model=model,
        max_tokens=1600,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is today's computed data object. Write the briefing using only "
                    "these numbers.\n\n```json\n" + payload + "\n```"
                ),
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
