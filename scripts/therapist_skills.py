
CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly and uptight. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
How effectively does the therapist use guided discovery techniques to facilitate client self-reflection and insight?

[criteria]
Score 0: Therapist relied primarily on debate, persuasion, or "lecturing." Therapist seemed to be "cross-examining" patient, putting the patient on the defensive, or forcing his/her point of view on the patient.
Score 2: Therapist relied too heavily on persuasion and debate, rather than guided discovery. However, therapist's style was supportive enough that patient did not seem to feel attacked or defensive.
Score 4: Therapist, for the most part, helped patient see new perspectives through guided discovery (e.g., examining evidence, considering alternatives, weighing advantages and disadvantages) rather than through debate. Used questioning appropriately.
Score 6: Therapist was especially adept at using guided discovery during the session to explore problems and help patient draw his/her own conclusions. Achieved an excellent balance between skillful questioning and other modes of intervention.

Do not forget to give a score strictly and uptight.
[Output]
"""

CBT_SPECIFIC_FOCUS = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
How well does the therapist identify and address the client’s key cognitions or behaviors that need change?

[criteria]
Score 0: Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.
Score 2: Therapist used appropriate techniques to elicit cognitions or behaviors; however, therapist had difficulty finding a focus or focused on cognitions/behaviors that were irrelevant to the patient’s key problems.
Score 4: Therapist focused on specific cognitions or behaviors relevant to the target problem. However, therapist could have focused on more central cognitions or behaviors that offered greater promise for progress.
Score 6: Therapist very skillfully focused on key thoughts, assumptions, behaviors, etc. that were most relevant to the problem area and offered considerable promise for progress. 
"""

CBT_SPECIFIC_STRATEGY = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
How appropriate and coherent is the therapist's strategy for promoting change in the client's problematic behaviors or thoughts?

[criteria]
Score 0: Therapist did not select cognitive-behavioral techniques.
Score 2: Therapist selected cognitive-behavioral techniques; however, either the overall strategy for bringing about change seemed vague or did not seem promising in helping the patient.
Score 4: Therapist seemed to have a generally coherent strategy for change that showed reasonable promise and incorporated cognitive-behavioral techniques.
Score 6: Therapist followed a consistent strategy for change that seemed very promising and incorporated the most appropriate cognitive-behavioral techniques.
"""

GEN_UNDERSTANDING = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. 
If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
How accurately does the therapist demonstrate understanding of the client's issues and concerns?

[criteria]
Score 0: Therapist repeatedly failed to understand what the patient explicitly said and thus consistently missed the point. Poor empathic skills.
Score 2: Therapist was usually able to reflect or rephrase what the patient explicitly said, but repeatedly failed to respond to more subtle communication. Limited ability to listen and empathize.
Score 4: Therapist generally seemed to grasp the patient’s “internal reality” as reflected by both what the patient explicitly said and what the patient communicated in more subtle ways. Good ability to listen and empathize.
Score 6: Therapist seemed to understand the patient’s “internal reality” thoroughly and was adept at communicating this understanding through appropriate verbal and non-verbal responses to the patient (e.g., the tone of the therapist’s response conveyed a sympathetic understanding of the client’s “message”). Excellent listening and empathic skills.
"""

GEN_INTERPERSONAL = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
How effective is the therapist in maintaining a positive and therapeutic relationship with the client?

[Criteria]
Score 0:Therapist had poor interpersonal skills. Seemed hostile, demeaning, or in some other way destructive to the patient.
Score 2: Therapist did not seem destructive, but had significant interpersonal problems. At times, therapist appeared unnecessarily impatient, aloof, insincere or had difficulty conveying confidence and competence.
Score 4: Therapist displayed a satisfactory degree of warmth, concern, confidence, genuineness, and professionalism. No significant interpersonal problems.
Score 6: Therapist displayed optimal levels of warmth, concern, confidence, genuineness, and professionalism, appropriate for this particular patient in this session.
"""

GEN_COLLABORATION = """
I want you to act as an evaluator. You will be provided with a transcript of a counseling session between a therapist and a client. Your task is to assess the therapist based on the given criteria. If you believe the therapist falls between two of the descriptors, select the intervening odd number (1, 3, 5). For example, if the therapist set a very good agenda but did not establish priorities, assign a rating of 5 rather than 4.

Please follow these steps:

1. Read the counseling session transcript carefully.
2. Review the evaluation questions and criteria provided below.
3. Assign a score based on the criteria, grading very strictly. If there is any deficiency, no matter how minor, assign a score of 4 or lower.
4. Output the score and the explanation, separated by a comma. Do not add any prefix.

[Counseling conversation]
{conversation}

[Evaluation Question]
To what extent does the therapist engage the client in collaborative goal-setting and decision-making?

[Criteria]
Score 0: Therapist did not attempt to set up a collaboration with patient.
Score 2: Therapist attempted to collaborate with patient, but had difficulty either defining a problem that the patient considered important or establishing rapport.
Score 4: Therapist was able to collaborate with patient, focus on a problem that both patient and therapist considered important, and establish rapport.
Score 6: Collaboration seemed excellent; therapist encouraged patient as much as possible to take an active role during the session (e.g., by offering choices) so they could function as a “team”.
"""


