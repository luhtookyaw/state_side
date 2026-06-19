"""Generate turn-level reward-model data from AIM therapist candidates."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aim_therapist import (  # noqa: E402
    AIMTherapist,
    render_prompt,
    stage_for_openness,
)
from chat_runtime import (  # noqa: E402
    DEFAULT_DATASET,
    ROOT_DIR,
    Turn,
    call_model,
    choose_patient,
    create_openai_client,
    load_dataset,
    load_environment,
    load_text,
    openai_model,
)
from client import (  # noqa: E402
    DEFAULT_CLIENT_PROMPT,
    DEFAULT_OPENNESS_JUDGE_PROMPT,
    INITIAL_OPENNESS_LEVEL,
    MODE_SETTINGS,
    OpennessJudge,
    SimulatedClient,
)
from generate_all_conversations import (  # noqa: E402
    DEFAULT_PATIENT_IDS_FILE,
    filter_patients,
    load_patient_ids,
    safe_patient_id,
)
from simulate_conversation import clamp_openness_transition  # noqa: E402


DEFAULT_REWARD_JUDGE_PROMPT = ROOT_DIR / "prompts" / "aim_therapist" / "reward_judge.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reward_data"
DEFAULT_REWARD_JUDGE_MODEL = "gpt-4o"
REWARD_SCORE_KEYS = (
    "therapeutic_alliance",
    "understanding_empathy",
    "collaboration_structure",
    "cbt_skill_change_strategy",
    "motivational_interviewing_readiness",
)
MODES = ("easy", "normal", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reward-model training data by scoring each AIM therapist "
            "candidate after a simulated one-step client response."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to patient scenario JSON dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL training data and conversation traces.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Difficulty modes to generate. Defaults to easy normal hard.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=30,
        help="Number of therapist-client exchange turns per conversation.",
    )
    parser.add_argument(
        "--patient-id",
        help="Generate data for one patient id, e.g. 1-1.",
    )
    parser.add_argument(
        "--patient-ids-file",
        type=Path,
        default=DEFAULT_PATIENT_IDS_FILE,
        help=(
            "Optional text file containing patient IDs to exclude, one per line. "
            "Blank lines and lines starting with # are ignored."
        ),
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        help="Optional limit on number of patients after filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when choosing a patient if --patient-id is omitted.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name for therapist, client, and reward judge.",
    )
    parser.add_argument(
        "--openness-judge-model",
        help=(
            "OpenAI model name for openness judging. Defaults to "
            "OPENNESS_JUDGE_MODEL, then --model."
        ),
    )
    parser.add_argument(
        "--reward-judge-model",
        default=DEFAULT_REWARD_JUDGE_MODEL,
        help=f"OpenAI model name for reward judging. Defaults to {DEFAULT_REWARD_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for therapist and client generation.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for reward and openness judges.",
    )
    parser.add_argument(
        "--client-prompt",
        type=Path,
        default=DEFAULT_CLIENT_PROMPT,
        help="Path to client prompt template.",
    )
    parser.add_argument(
        "--openness-judge-prompt",
        type=Path,
        default=DEFAULT_OPENNESS_JUDGE_PROMPT,
        help="Path to openness judge prompt template.",
    )
    parser.add_argument(
        "--reward-judge-prompt",
        type=Path,
        default=DEFAULT_REWARD_JUDGE_PROMPT,
        help="Path to reward judge prompt template.",
    )
    parser.add_argument(
        "--max-context-turns",
        type=int,
        default=8,
        help=(
            "Number of recent turns to include in reward-model input and reward "
            "judging. Use 0 for full conversation history."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSONL and trace files.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print selected candidate and score for each turn.",
    )
    return parser.parse_args()


def format_history(turns: list[Turn], max_turns: int) -> str:
    if not turns:
        return "No previous turns."
    selected_turns = turns if max_turns == 0 else turns[-max_turns:]
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in selected_turns)


def parse_reward_scores(raw: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in REWARD_SCORE_KEYS:
        pattern = rf"^{re.escape(key)}\s*:\s*(10|[1-9])(?:\.0+)?\s*$"
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            raise ValueError(f"Could not parse reward score {key!r} from: {raw!r}")
        scores[key] = float(match.group(1))
    return scores


def final_reward_score(scores: dict[str, float]) -> float:
    return sum(scores[key] for key in REWARD_SCORE_KEYS) / len(REWARD_SCORE_KEYS)


class RewardJudge:
    """Scores a candidate branch with the reward_judge prompt."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path,
        max_context_turns: int,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)
        self.max_context_turns = max_context_turns

    def judge(self, conversation: list[Turn]) -> dict[str, Any]:
        prompt = render_prompt(
            self.template,
            conversation_history=format_history(conversation, self.max_context_turns),
        )
        raw = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=220,
        )
        scores = parse_reward_scores(raw)
        return {
            "scores": scores,
            "final_score": final_reward_score(scores),
            "raw": raw,
        }


def load_patients_for_args(args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    dataset = load_dataset(args.dataset)
    if args.patient_id:
        return [choose_patient(dataset, args.patient_id, rng)]

    excluded_patient_ids = (
        set(load_patient_ids(args.patient_ids_file)) if args.patient_ids_file else set()
    )
    if excluded_patient_ids:
        filter_patients(dataset, list(excluded_patient_ids))
    patients = [
        patient
        for patient in dataset
        if str(patient.get("id")) not in excluded_patient_ids
    ]
    if args.max_clients is not None:
        patients = patients[: args.max_clients]
    return patients


def reward_record(
    *,
    patient: dict[str, Any],
    mode: str,
    turn_number: int,
    openness_level: int,
    stage: str,
    base_conversation: list[Turn],
    candidate: dict[str, str],
    candidate_metadata: dict[str, Any] | None,
    branch_client_text: str,
    reward: dict[str, Any],
    selected: bool,
    max_context_turns: int,
) -> dict[str, Any]:
    candidate_conversation = [
        *base_conversation,
        Turn("Therapist", candidate["response"]),
    ]
    judged_conversation = [
        *candidate_conversation,
        Turn("Client", branch_client_text),
    ]
    return {
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "mode": mode,
        "turn": turn_number,
        "openness_level": openness_level,
        "stage": stage,
        "candidate_id": candidate["id"],
        "candidate_response": candidate["response"],
        "candidate_metadata": candidate_metadata,
        "branch_client_response": branch_client_text,
        "reward_scores": reward["scores"],
        "final_score": reward["final_score"],
        "selected_for_continuation": selected,
        "reward_judge_raw": reward["raw"],
        "reward_model_input": format_history(candidate_conversation, max_context_turns),
        "judged_conversation": format_history(judged_conversation, max_context_turns),
    }


def choose_best_candidate(scored_records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        scored_records,
        key=lambda record: (
            record["final_score"],
            record["reward_scores"]["therapeutic_alliance"],
            record["reward_scores"]["motivational_interviewing_readiness"],
            -int(str(record["candidate_id"]).removeprefix("candidate_") or 0),
        ),
    )


def simulate_reward_conversation(
    *,
    patient: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    openai_client: Any,
    model: str,
    openness_judge_model: str,
    reward_judge_model: str,
) -> dict[str, Any]:
    therapist = AIMTherapist(openai_client, model, args.temperature)
    client = SimulatedClient(
        patient,
        mode,
        openai_client,
        model,
        args.temperature,
        prompt_path=args.client_prompt,
    )
    openness_judge = OpennessJudge(
        openai_client,
        openness_judge_model,
        args.judge_temperature,
        prompt_path=args.openness_judge_prompt,
    )
    reward_judge = RewardJudge(
        openai_client,
        reward_judge_model,
        args.judge_temperature,
        args.reward_judge_prompt,
        args.max_context_turns,
    )

    mode_settings = MODE_SETTINGS[mode]
    openness_level = INITIAL_OPENNESS_LEVEL
    openness_judge_interval = mode_settings["openness_judge_interval"]
    conversation: list[Turn] = []
    training_records: list[dict[str, Any]] = []
    turn_traces: list[dict[str, Any]] = []
    openness_judgments: list[dict[str, Any]] = []

    for turn_number in range(1, args.turns + 1):
        openness_level_before_turn = openness_level

        if turn_number == 1:
            therapist_text = therapist.opening(patient)
            therapist_turn = Turn("Therapist", therapist_text)
            conversation.append(therapist_turn)
            client_text = client.reply(conversation, openness_level_before_turn)
            client_turn = Turn("Client", client_text)
            conversation.append(client_turn)
            turn_traces.append(
                {
                    "turn": turn_number,
                    "openness_level": openness_level_before_turn,
                    "stage": "pre-contemplation",
                    "opening": True,
                    "therapist": therapist_text,
                    "client": client_text,
                }
            )
        else:
            stage = stage_for_openness(openness_level_before_turn)
            candidates, candidate_metadata = therapist.generate_candidates(
                patient,
                conversation,
                openness_level_before_turn,
            )
            scored_records: list[dict[str, Any]] = []

            for candidate in candidates:
                branch_conversation = [
                    *conversation,
                    Turn("Therapist", candidate["response"]),
                ]
                branch_client_text = client.reply(
                    branch_conversation,
                    openness_level_before_turn,
                )
                judged_branch = [
                    *branch_conversation,
                    Turn("Client", branch_client_text),
                ]
                reward = reward_judge.judge(judged_branch)
                record = reward_record(
                    patient=patient,
                    mode=mode,
                    turn_number=turn_number,
                    openness_level=openness_level_before_turn,
                    stage=stage,
                    base_conversation=conversation,
                    candidate=candidate,
                    candidate_metadata=candidate_metadata.get(candidate["id"]),
                    branch_client_text=branch_client_text,
                    reward=reward,
                    selected=False,
                    max_context_turns=args.max_context_turns,
                )
                scored_records.append(record)

            best_record = choose_best_candidate(scored_records)
            for record in scored_records:
                record["selected_for_continuation"] = (
                    record["candidate_id"] == best_record["candidate_id"]
                )
            training_records.extend(scored_records)

            conversation.append(Turn("Therapist", best_record["candidate_response"]))
            conversation.append(Turn("Client", best_record["branch_client_response"]))

            turn_traces.append(
                {
                    "turn": turn_number,
                    "openness_level": openness_level_before_turn,
                    "stage": stage,
                    "candidates": scored_records,
                    "selected_candidate_id": best_record["candidate_id"],
                    "selected_final_score": best_record["final_score"],
                    "therapist": best_record["candidate_response"],
                    "client": best_record["branch_client_response"],
                }
            )

            if args.print:
                print(
                    f"{patient.get('id')} {mode} turn {turn_number}: "
                    f"{best_record['candidate_id']} score="
                    f"{best_record['final_score']:.2f}"
                )

        openness_judgment: dict[str, Any] | None = None
        if turn_number % openness_judge_interval == 0:
            openness_judgment = openness_judge.judge(conversation, turn_number)
            raw_openness_level = openness_judgment["openness_level"]
            openness_level = clamp_openness_transition(
                raw_openness_level,
                openness_level_before_turn,
            )
            openness_judgment["raw_openness_level"] = raw_openness_level
            openness_judgment["openness_level"] = openness_level
            openness_judgments.append(openness_judgment)
            turn_traces[-1]["openness_judgment"] = openness_judgment

    return {
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "mode": mode,
        "turns": args.turns,
        "model": model,
        "openness_judge_model": openness_judge_model,
        "reward_judge_model": reward_judge_model,
        "initial_openness_level": INITIAL_OPENNESS_LEVEL,
        "final_openness_level": openness_level,
        "openness_judge_interval": openness_judge_interval,
        "client_type": client.client_type,
        "training_records": training_records,
        "openness_judgments": openness_judgments,
        "conversation": [{"speaker": turn.speaker, "text": turn.text} for turn in conversation],
        "turn_traces": turn_traces,
    }


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_trace(path: Path, trace: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.turns < 1:
        raise SystemExit("--turns must be at least 1.")
    if args.max_context_turns < 0:
        raise SystemExit("--max-context-turns must be 0 or greater.")

    load_environment()
    rng = random.Random(args.seed)
    model = args.model or openai_model()
    openness_judge_model = args.openness_judge_model or model
    reward_judge_model = args.reward_judge_model
    patients = load_patients_for_args(args, rng)
    openai_client = create_openai_client()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "reward_training_data.jsonl"
    trace_dir = args.output_dir / "traces"

    if jsonl_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists. Use --overwrite to replace: {jsonl_path}")
    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    total_sessions = len(patients) * len(args.modes)
    completed_sessions = 0
    total_records = 0

    for patient in patients:
        patient_id = patient.get("id")
        for mode in args.modes:
            completed_sessions += 1
            print(
                f"[generate] {completed_sessions}/{total_sessions} "
                f"patient={patient_id} mode={mode}",
                flush=True,
            )
            trace = simulate_reward_conversation(
                patient=patient,
                mode=mode,
                args=args,
                openai_client=openai_client,
                model=model,
                openness_judge_model=openness_judge_model,
                reward_judge_model=reward_judge_model,
            )
            records = trace["training_records"]
            append_jsonl(jsonl_path, records)
            total_records += len(records)

            trace_path = (
                trace_dir
                / mode
                / f"session_{safe_patient_id(patient_id)}_{mode}_aim_reward.json"
            )
            write_trace(trace_path, trace)

    print(f"Saved {total_records} reward records to {jsonl_path}")
    print(f"Saved conversation traces to {trace_dir}")


if __name__ == "__main__":
    main()
