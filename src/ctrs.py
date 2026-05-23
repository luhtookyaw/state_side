from collections.abc import Callable
from typing import Any
import json
import re


def _collapse_string_newlines(text: str) -> str:
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace('\n', ' ')
    return ''.join(parts)


def _parse_llm_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    fixed = re.sub(r'([\"\d\}\]])\s*\n(\s*\")', r'\1,\n\2', fixed)
    fixed = _collapse_string_newlines(fixed)

    return json.loads(fixed)


CTRS_SYSTEM_PROMPT = """
You are an expert CBT supervisor using the Cognitive Therapy Rating Scale (CTRS) to evaluate a therapist's performance in a single session. Rate only the therapist's observable behavior in this session, taking into account the difficulty of the patient.

Use the CTRS 0-6 scale:
0 = Poor, 1 = Barely adequate, 2 = Mediocre, 3 = Between mediocre and satisfactory, 4 = Satisfactory, 5 = Between good and very good, 6 = Excellent.

For each CTRS item, you must:
- Assign one integer score from 0-6 (no half points).
- Give a brief 1-2 sentence justification referencing specific therapist behaviors.
- Ignore items only if absolutely impossible to rate; in that case, explain why.

Return output as strict JSON with this schema:
{
  "part_I": {
    "1_agenda": {"score": <0-6>, "justification": "<text>"},
    "2_feedback": {"score": <0-6>, "justification": "<text>"},
    "3_understanding": {"score": <0-6>, "justification": "<text>"},
    "4_interpersonal_effectiveness": {"score": <0-6>, "justification": "<text>"},
    "5_collaboration": {"score": <0-6>, "justification": "<text>"},
    "6_pacing_and_time_use": {"score": <0-6>, "justification": "<text>"}
  },
  "part_II": {
    "7_guided_discovery": {"score": <0-6>, "justification": "<text>"},
    "8_key_cognitions_behaviors": {"score": <0-6>, "justification": "<text>"},
    "9_strategy_for_change": {"score": <0-6>, "justification": "<text>"},
    "10_cb_techniques_application": {"score": <0-6>, "justification": "<text>"},
    "11_homework": {"score": <0-6>, "justification": "<text>"}
  },
  "total_score": <int>,
  "global_comments": "<2-5 sentence qualitative summary>"
}

CTRS item anchors and definitions (for reference):
1. Agenda: quality of setting and following a mutually agreed, specific agenda with prioritized target problems; efficient use of available time.
2. Feedback: how consistently the therapist checks the client's understanding and reactions and adjusts interventions accordingly.
3. Understanding: accuracy and depth of empathic understanding of the client's explicit and subtle communications.
4. Interpersonal effectiveness: warmth, concern, confidence, genuineness, and professionalism; absence of hostility, aloofness, or destructiveness.
5. Collaboration: success in establishing a cooperative, "team" relationship and focusing on problems important to both therapist and client.
6. Pacing and efficient use of time: degree to which the therapist structures the session and keeps discussion focused and productive.
7. Guided discovery: use of skillful questioning and collaborative exploration rather than debate or persuasion to help clients reach their own conclusions.
8. Focusing on key cognitions or behaviors: extent to which the therapist identifies and works with central thoughts, assumptions, images, and behaviors relevant to target problems.
9. Strategy for change: coherence and promise of the overall CBT change plan selected for this session.
10. Application of cognitive-behavioral techniques: skillfulness in implementing CBT methods (e.g., cognitive restructuring, behavioral experiments, exposure, activity scheduling).
11. Homework: review and assignment of homework that is clear, appropriate, and linked to session content.
"""


def evaluate_ctrs_transcript(
    transcript: str, model_call: Callable[[str, int], str]
) -> dict[str, Any]:
    prompt = f"""
{CTRS_SYSTEM_PROMPT}

Use the Cognitive Therapy Rating Scale (CTRS) to evaluate the therapist in the following session.

Session content (Full Session):
{transcript}

IMPORTANT: Respond with valid JSON only. Each justification must be a single line — do not use line breaks inside string values.
"""
    raw = model_call(prompt, 2200)
    try:
        return _parse_llm_json(raw)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}
