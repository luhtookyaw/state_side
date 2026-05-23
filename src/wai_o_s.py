from collections.abc import Callable
from typing import Any
import json
import re

RUBRIC = """
Q1: There are doubts or a lack of understanding about what participants are trying to accomplish in therapy.
Dimension: Goal
Evaluation:
1 = The counselor or the client explicitly mentions the counseling goals and works around the established objectives, such as understanding information related to the goals and methods to achieve them. The relevance of the dialogue to the goals is evident for both the counselor and the client. They may discuss the goals to acknowledge or comment on the usefulness of the therapeutic process.
2 = The counselor and the client do not explicitly mention the goals but are working towards a common objective. The counselor addresses the client's concerns immediately and adjusts the therapeutic process to meet the client's needs. The client is satisfied with the progress made.
3 = There is no evidence to suggest that the counselor and the client have established consistent counseling goals, or there is an equal level of confusion and understanding regarding the goals.
4 = There is disagreement between the counselor and the client regarding counseling goals. While there may be some communication between both parties, the counselor's specific tasks or interventions may be questioned or resisted by the client. The counseling may need to be paused multiple times to adjust the goals. The client may express overall dissatisfaction with the counseling.
5 = The counselor and the client have clearly identified different goals, and there are disagreements in the order of issues and solutions in therapy. This inconsistency may lead the client to express strong dissatisfaction with the overall counseling process and goals.

Q2: The client and therapist are working on mutually agreed upon goals.
Dimension: Goal
Evaluation:
1 = The shift of topics often occurs abruptly, usually without mutual agreement from both parties. This frequent topic shift may result from one party interrupting or disregarding the other's statements. At this stage, significant conflicts exist between the counselor and the client regarding the appropriateness, definition, and boundaries of the goals.
2 = Topics may shift before resolution or conclusion, but the transition typically moves from one relevant topic to another related or less related one. This shift can be initiated by either the counselor or the client.
3 = There may be some ambiguity or uncertainty between the counselor and the client regarding session goals. The current stage of communication lacks clear evidence that both parties have reached a common understanding or collaboration.
4 = The counselor and the client have made some progress through discussing relevant topics, but there may still be a small amount of disagreement or areas that need further exploration.
5 = The counselor and the client have achieved complete agreement on goals through in-depth, targeted discussions, and have had highly productive discussions on multiple related topics.

Q3: The client and therapist have different ideas about what the client's real problems are.
Dimension: Goal
Evaluation:
1 = The counselor and the client have a very clear and consistent understanding of the client's issues and goals. There is a strong consensus on problem resolution.
2 = The counselor and the client have a certain level of consensus on the client's issues and goals. Both parties are making efforts to understand each other.
3 = In the communication regarding the client's issues, there is no clear evidence of agreement or disagreement.
4 = There is some disagreement between the counselor and the client regarding the client's issues.
5 = There is evident conflict and disagreement between the counselor and the client in defining and addressing the client's issues.

Q4: The client and therapist have established a good understanding of the changes that would be good for the client.
Dimension: Goal
Evaluation:
1 = There are clear misunderstandings and disagreements between the counselor and the client in the process of change.
2 = The client may have doubts or uncertainties in the process of change.
3 = The counselor and the client have a neutral attitude towards the process and goals of change in the conversation.
4 = Both the counselor and the client are aware of changes that would benefit the client.
5 = There is strong consistency and clarity between the counselor and the client regarding the client's goals and how to achieve them.

Q5: There is agreement about the steps taken to help improve the client's situation.
Dimension: Approach
Evaluation:
1 = The client directly expresses that the tasks and goals are inappropriate and generally disagrees with homework or tasks during the session. The client refuses to engage in tasks.
2 = The client hesitates to explore and does not follow the counselor's guidance in the change process. The client withdraws from the counselor.
3 = There is no clear consensus or disagreement between the counselor and the client regarding therapy tasks.
4 = The client shows a clear interest and involvement in therapy tasks. The client participates and follows the exploration process.
5 = The counselor and client strongly agree on the therapeutic approach and actively cooperate on steps and tasks.

Q6: There is agreement about the usefulness of the current activity in therapy.
Dimension: Approach
Evaluation:
1 = The client repeatedly argues against tasks. The client refuses to participate, claiming that it is pointless.
2 = The client does not actively engage in the session tasks, although he/she may not openly question the usefulness of the tasks.
3 = There is no clear evidence about whether they have reached an agreement or disagreement.
4 = The client actively participates in and is committed to therapy tasks, showing no skepticism about their effectiveness.
5 = The counselor and the client have a strong and clear agreement on the client's goals and how to achieve them.

Q7: There is agreement on what is important for the client to work on.
Dimension: Approach
Evaluation:
1 = There is a clear disagreement and opposition between the counselor and the client regarding the current focus.
2 = The counselor and the client have some disagreement about the content and direction of therapy.
3 = There are no clear signs of agreement or disagreement regarding the themes or issues of therapy.
4 = The client and the counselor respond to each other's focus and needs to some extent.
5 = The counselor and the client are highly actively engaged, thoroughly exploring each other's issues and responding explicitly.

Q8: The client believes that the way they are working with his/her problem is correct.
Dimension: Approach
Evaluation:
1 = The client holds evident doubts and aversions towards the counseling process, frequently engaging in arguments with the counselor.
2 = The counselor and the client sometimes have conflicting opinions, but they seem to cooperate in certain parts.
3 = The client maintains a neutral stance toward the therapy process and methods.
4 = The client partially agrees with certain aspects of therapy tasks.
5 = The client is satisfied and excited about the counselor's methods and approach to problem-solving.

Q9: There is a mutual liking between the client and therapist.
Dimension: Affective Bond
Evaluation:
1 = There is evident animosity, hostility, or indifference between the counselor and the client.
2 = Although there is no direct hostility, there is noticeable tension and distance in the relationship.
3 = There are no clear signs of warmth or coldness in the relationship.
4 = In the majority of the sessions, the counselor and the client have positive interactions.
5 = Throughout the therapy process, both consistently demonstrate a deep care for each other and provide positive feedback.

Q10: The client feels confident in the therapist's ability to help the client.
Dimension: Affective Bond
Evaluation:
1 = The client expresses minimal or no hope for the therapy outcomes.
2 = The client harbors doubts about the therapist, the therapy process, or the anticipated outcomes.
3 = The client holds a neutral stance regarding the therapist's capabilities.
4 = The client expresses a certain level of confidence in the therapist's abilities.
5 = The client consistently agrees with the therapist's reflections and interventions/guidance.

Q11: The client feels that the therapist appreciates him/her.
Dimension: Affective Bond
Evaluation:
1 = The client feels that the therapist is indifferent, inattentive, and unconcerned.
2 = The client harbors some doubts about whether the therapist genuinely cares.
3 = There is no clear evidence of strong positive or negative reactions from the client regarding the therapist's care.
4 = The therapist demonstrates acceptance, warmth, and empathy, and the client perceives this caring attitude.
5 = The client strongly senses the therapist's care and support, expressing gratitude for the relationship.

Q12: There is mutual trust between the client and therapist.
Dimension: Affective Bond
Evaluation:
1 = The client has significant mistrust towards the therapist, demonstrated by avoiding discussions on critical issues.
2 = There is a moderate level of mistrust between both parties.
3 = There are no clear signs of trust, but there are also no apparent behaviors indicating mistrust.
4 = The client is willing to disclose some personal concerns, and the therapist accepts the client's surface statements.
5 = The trust between both parties is deep enough that the client willingly shares deeper layers of privacy and issues.
"""

WAI_O_S_PROMPT = """
You are an expert supervisor for a therapy simulation. Your task is to evaluate the therapeutic alliance using the Observer-rated Short Version of the Working Alliance Inventory (WAI-O-S).

Check the RUBRIC below for the exact criteria for each score (1-5) for each of the 12 items.
IMPORTANT: Some items in the rubric describe a "Good" alliance with a score of 1 and "Bad" with 5 (Reverse Scored in the Rubric itself), while others are normal (5=Good).
Specifically:
- Q1: Rubric says 1 = "Explicitly mentions goals... relevance evident" (Good). 5 = "Clearly identified different goals... inconsistency" (Bad). So 1 is BEST.
- Q3: Rubric says 1 = "Clear and consistent understanding" (Good). 5 = "Evident conflict" (Bad). So 1 is BEST.
- Q5: Treat '5' as "Strong Agreement" (positive alliance).

Your output must be a JSON object containing:
1. "item_scores": A dictionary with keys "Q1" to "Q12". Each value should be an object:
   - "raw_score": The 1-5 score based exactly on the rubric description match.
   - "normalized_score": A score from 1-5 where 5 represents the STRONGEST POSITIVE ALLIANCE and 1 represents the WEAKEST.
     - For Q1 and Q3 (and any others where 1 is described as positive in the rubric): normalized_score = 6 - raw_score.
     - For others (Q2, Q4, Q6-Q12): normalized_score = raw_score.
   - "reason": Brief explanation.
2. "dimension_scores": Average of the NORMALIZED scores for each dimension:
   - "goal": Mean of Q1, Q2, Q3, Q4 (normalized)
   - "approach": Mean of Q5, Q6, Q7, Q8 (normalized)
   - "bond": Mean of Q9, Q10, Q11, Q12 (normalized)
3. "total_score": Mean of all 12 normalized scores.

"""


def _parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def evaluate_wai_o_s_transcript(
    transcript: str, model_call: Callable[[str, int], str]
) -> dict[str, Any]:
    prompt = (
        WAI_O_S_PROMPT
        + "\n\nRUBRIC:\n"
        + RUBRIC
        + "\n\nAnalyze this conversation:\n"
        + transcript
    )
    raw = model_call(prompt, 2400)
    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}
