from collections.abc import Callable
from typing import Any
import json
import re

MITI_SYSTEM_PROMPT = """
You are an expert supervisor in Motivational Interviewing (MI) using a MITI-style fidelity framework to evaluate a MI segment from a therapy session. Your task is to rate the therapist's MI-consistent behaviors and provide brief justifications.

Use the following 1-5 global rating scale for each dimension:
1 = Very low MI consistency
2 = Low MI consistency
3 = Moderate MI consistency
4 = Good MI consistency
5 = Excellent MI consistency

Global dimensions to rate:
- Partnership/Collaboration: How much the therapist works with the client as a team and avoids an expert, one-up stance.
- Empathy: How accurately and warmly the therapist understands and reflects the client's perspective.
- Cultivating Change Talk: How effectively the therapist evokes, strengthens, and responds to client language in favor of change.
- Softening Sustain Talk: How well the therapist avoids arguing, reduces defensiveness, and responds non-confrontationally to language in favor of the status quo.
- Direction: Whether the therapist gently guides the conversation toward a clear change goal while preserving client autonomy.

Also count the following observable behaviors in the segment:
- Open questions
- Closed questions
- Simple reflections
- Complex reflections (adding meaning or feeling)
- Affirmations
- Summaries
- MI-consistent behaviors (e.g., supporting autonomy, emphasizing control, asking permission)
- MI-inconsistent behaviors (e.g., confronting, directing without permission, arguing, warning, persuading without permission)

Note:
- For each global rating, provide a 1-3 sentence justification referring to specific therapist utterances or patterns (no need to quote verbatim; paraphrasing is fine).
- For behavior counts, provide integers based on the observable turns in the segment.
- Output must be valid JSON only, following the schema specified in the user message.
"""


def _parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def evaluate_miti_transcript(
    transcript: str, model_call: Callable[[str, int], str]
) -> dict[str, Any]:
    prompt = f"""
{MITI_SYSTEM_PROMPT}

Evaluate the following MI segment from a therapy/chatbot conversation. Focus only on turns where the speaker is the therapist or chatbot, not the client.

MI segment transcript:
{transcript}

Return your evaluation as strict JSON in the following structure:

{{
  "session_id": "session",
  "segment_type": "MI",
  "target_behavior": "General",
  "global_ratings": {{
    "partnership": {{"score": 1, "justification": "..."}},
    "empathy": {{"score": 1, "justification": "..."}},
    "cultivating_change_talk": {{"score": 1, "justification": "..."}},
    "softening_sustain_talk": {{"score": 1, "justification": "..."}},
    "direction": {{"score": 1, "justification": "..."}}
  }},
  "behavior_counts": {{
    "open_questions": 0,
    "closed_questions": 0,
    "simple_reflections": 0,
    "complex_reflections": 0,
    "affirmations": 0,
    "summaries": 0,
    "mi_consistent_behaviors": 0,
    "mi_inconsistent_behaviors": 0
  }},
  "overall_comments": "2-3 sentence summary of MI quality in this segment, including how well it supports readiness for CBT work."
}}

Rules:
- All scores must be integers 1-5.
- behavior_counts must be non-negative integers.
- If some behavior type does not occur, set its count to 0 and do not invent examples.
- Keep justifications concise.
"""
    raw = model_call(prompt, 1800)
    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}
