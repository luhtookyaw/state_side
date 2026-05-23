from collections.abc import Callable
from typing import Any
import json
import re

ALLIANCE_PROMPT = """
You are an expert supervisor for a therapy simulation. Your task is to evaluate the therapeutic alliance between the therapist and the patient based on the Working Alliance Inventory (WAI-O) - Observer form.

The WAI-O consists of 36 items, rated on a 7-point scale (1=Never, 7=Always).

IMPORTANT:
- High scores (7) should ALWAYS indicate a POSITIVE/STRONG alliance. If an item is negatively worded (e.g., "There is a sense of discomfort"), you must REVERSE SCORE it implicitly so that a 7 means "No discomfort" (i.e. strong alliance) and 1 means "High discomfort".
- Assume that this is a real therapy session where the goal is to help the client with genuine problems, even though it is simulated.
- Focus on what is observable in the interaction: verbal behavior, emotional tone, degree of collaboration, and evidence of agreement or disagreement.
- You do not need to judge diagnostic accuracy or treatment outcomes. Your focus is only on the quality of the working alliance (bond, agreement on goals, agreement on tasks) as expressed in the interaction.
- Rate each item based only on what appears in this single session.
- If an item is difficult to judge, use your best inference from the observable interaction and choose the most appropriate point on the scale.

Here are the 36 items:
Bond Subscale:
1. There is a sense of discomfort in the relationship. (Reverse)
5. There is good understanding between the client and therapist.
8. There is a mutual liking between the client and therapist.
17. The client is aware that the therapist is genuinely concerned for his/her welfare.
19. The client and the therapist respect each other.
20. The client feels that the therapist is not totally honest about his/her feelings toward her/him. (Reverse)
23. The client feels that the therapist appreciates him/her as a person.
26. There is mutual trust between the client and therapist.
28. Both the client and therapist see their relationship as important to the client.
29. The client fears that if he/she says or does the wrong things, the therapist will stop working with him/her. (Reverse)
35. The client believes that the way they are working with his/her problem is correct.
36. The client feels that the therapist respects and cares about the client, even when the client does things the therapist does not approve of.

Goal Subscale:
3. There is concern about the outcome of the sessions. (Reverse)
6. There is a shared perception of the client's goals in therapy.
10. There is disagreement about the goals of the session. (Reverse)
12. There are doubts or a lack of understanding about what participants are trying to accomplish in therapy. (Reverse)
14. There is a mutual perception that the goals of the sessions are important for the client.
16. There is agreement that what the client and therapist are doing in therapy will help the client to accomplish the changes he/she wants.
18. There is clarity about what the therapist wants the client to do.
22. The client and therapist are working on mutually agreed upon goals.
24. There is agreement on what is important for the client to work on.
27. The client and therapist have different ideas about what the client's real problems are. (Reverse)
30. The client and therapist collaborated on setting the goals for the session.
32. The client and therapist have established a good understanding of the changes that would be good for the client.

Task Subscale:
2. There is agreement about the steps taken to help improve the client's situation.
4. There is agreement about the usefulness of the current activity in therapy.
7. There is a sense of confusion between the client and therapist about what they are doing in therapy. (Reverse)
9. There is a need to clarify the purpose of the sessions. (Reverse)
11. There is a perception that the time spent in therapy is not spent efficiently. (Reverse)
13. There is agreement about what client's responsibilities are in therapy.
15. There is the perception that what the therapist and client are doing in therapy is unrelated to the client's current concerns. (Reverse)
21. The client feels confident in the therapist's ability to help the client.
25. As a result of these sessions there is clarity about how the client might be able to change.
31. The client is frustrated with what he/she is being asked to do in the therapy. (Reverse)
33. The therapy process does not make sense to the client. (Reverse)
34. The client doesn't know what to expect as the result of therapy. (Reverse)

Assign a score from 1 to 7 for each of the 36 items based on the conversation history.
Calculate the sum for each subscale:
- Bond Score (Sum of 12 items)
- Goal Score (Sum of 12 items)
- Task Score (Sum of 12 items)
- Total Score (Sum of all 36 items)

Additionally, provide a brief qualitative analysis (1-2 sentences) for each subscale explaining why you assigned those scores, citing specific interactions if possible.

Return ONLY a JSON object with this structure:
{
    "bond_score": integer,
    "bond_analysis": string,
    "goal_score": integer,
    "goal_analysis": string,
    "task_score": integer,
    "task_analysis": string,
    "total_score": integer,
    "item_scores": {
        "1": integer,
        "2": integer,
        ...
        "36": integer
    }
}
"""


def _parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def evaluate_wai_o_transcript(
    transcript: str, model_call: Callable[[str, int], str]
) -> dict[str, Any]:
    prompt = f"""
{ALLIANCE_PROMPT}

Analyze the following therapy session conversation and evaluate the Therapeutic Alliance.

Conversation History:
{transcript}
"""
    raw = model_call(prompt, 3600)
    try:
        scores = _parse_json_response(raw)
    except json.JSONDecodeError:
        scores = {"parse_error": True, "raw": raw}

    return {
        "observers": [
            {
                "name": "Observer",
                "scores": scores,
            }
        ]
    }
