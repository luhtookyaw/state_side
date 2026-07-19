"""Generate turn-level reward-model data from AIM therapist candidates."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
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
DEFAULT_BEAM_WIDTH = 2
DEFAULT_DISCOUNT_FACTOR = 0.9
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
            "Generate D1, D2, or D3 reward-model data from AIM therapist "
            "candidate trajectories."
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
        "--output-file",
        type=Path,
        help=(
            "Optional JSONL output path. Defaults to reward_training_data.jsonl "
            "for D1 and reward_training_data_dN.jsonl for deeper lookahead."
        ),
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Lookahead depth and dataset stage to generate. Defaults to 1 (D1).",
    )
    parser.add_argument(
        "--reward-model",
        type=Path,
        help="Hugging Face reward-model checkpoint required for D2 and D3.",
    )
    parser.add_argument(
        "--reward-model-max-length",
        type=int,
        default=512,
        help="Maximum tokenizer length for local reward-model inference.",
    )
    parser.add_argument(
        "--reward-model-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for local reward-model inference. Defaults to auto.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help="Number of trajectories retained between lookahead depths.",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=DEFAULT_DISCOUNT_FACTOR,
        help="Gamma used for cumulative predicted rewards. Defaults to 0.9.",
    )
    parser.add_argument(
        "--random-terminal-candidates",
        type=int,
        default=1,
        help="Additional random terminal trajectories judged for D2/D3.",
    )
    parser.add_argument(
        "--low-ranked-terminal-candidates",
        type=int,
        default=1,
        help="Additional lowest-ranked terminal trajectories judged for D2/D3.",
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
        "--max-context-pairs",
        type=int,
        default=4,
        help=(
            "Number of recent Therapist/Client pairs to include in reward judging and "
            "judged-conversation traces. Use 0 for full conversation history."
        ),
    )
    parser.add_argument(
        "--reward-model-context-pairs",
        type=int,
        default=4,
        help=(
            "Number of preceding Therapist/Client pairs to include before the "
            "candidate Therapist response in reward_model_input. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSONL and trace files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run by appending missing sessions and skipping "
            "patient/mode sessions whose trace file already exists."
        ),
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


def format_paired_history(turns: list[Turn], context_pairs: int) -> str:
    if not turns:
        return "No previous turns."
    selected_turns = turns if context_pairs == 0 else turns[-context_pairs * 2 :]
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in selected_turns)


def format_reward_model_input(turns: list[Turn], context_pairs: int) -> str:
    if not turns:
        return "No previous turns."
    if context_pairs == 0:
        selected_turns = turns
    else:
        final_turn = turns[-1:]
        preceding_turns = turns[:-1]
        selected_turns = [
            *preceding_turns[-context_pairs * 2 :],
            *final_turn,
        ]
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
        max_context_pairs: int,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)
        self.max_context_pairs = max_context_pairs

    def judge(self, conversation: list[Turn]) -> dict[str, Any]:
        prompt = render_prompt(
            self.template,
            conversation_history=format_paired_history(
                conversation,
                self.max_context_pairs,
            ),
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


class RewardModelScorer:
    """Run batched scalar inference from a local Hugging Face checkpoint."""

    def __init__(self, checkpoint: Path, max_length: int, device_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "D2/D3 generation requires torch and transformers. Install the "
                "training dependencies from requirements.txt."
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--reward-model-device cuda requested, but CUDA is unavailable.")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.model.eval()
        self.output_scale = self._load_output_scale(checkpoint)

    @staticmethod
    def _load_output_scale(checkpoint: Path) -> float:
        metric_paths = (checkpoint / "metrics.json", checkpoint.parent / "metrics.json")
        for path in metric_paths:
            if not path.exists():
                continue
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metrics.get("normalize_target") is True:
                return 10.0
        return 1.0

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits.reshape(-1)
        return [float(value) * self.output_scale for value in logits.cpu().tolist()]


@dataclass
class SearchTrajectory:
    trajectory_id: str
    conversation: list[Turn]
    steps: list[dict[str, Any]]
    cumulative_reward: float
    cbt_techniques: list[str]

    @property
    def latest_step(self) -> dict[str, Any]:
        return self.steps[-1]


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
    max_context_pairs: int,
    reward_model_context_pairs: int,
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
        "dataset_stage": "D1",
        "lookahead_depth": 1,
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
        "reward_model_input": format_reward_model_input(
            candidate_conversation,
            reward_model_context_pairs,
        ),
        "judged_conversation": format_paired_history(
            judged_conversation,
            max_context_pairs,
        ),
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


def cumulative_reward(steps: list[dict[str, Any]], gamma: float) -> float:
    return sum(
        (gamma ** (index - 1)) * float(step["predicted_reward"])
        for index, step in enumerate(steps, start=1)
    )


def trajectory_sort_key(trajectory: SearchTrajectory) -> tuple[float, str]:
    return (trajectory.cumulative_reward, trajectory.trajectory_id)


def complete_latest_client_response(
    trajectory: SearchTrajectory,
    client: SimulatedClient,
    openness_level: int,
) -> None:
    if trajectory.latest_step.get("client_response") is not None:
        return
    client_text = client.reply(trajectory.conversation, openness_level)
    trajectory.latest_step["client_response"] = client_text
    trajectory.conversation.append(Turn("Client", client_text))


def select_terminal_trajectories(
    trajectories: list[SearchTrajectory],
    *,
    top_count: int,
    random_count: int,
    low_count: int,
    rng: random.Random,
) -> list[tuple[SearchTrajectory, str]]:
    ranked = sorted(trajectories, key=trajectory_sort_key, reverse=True)
    selected: list[tuple[SearchTrajectory, str]] = []
    used_ids: set[str] = set()

    for trajectory in ranked[:top_count]:
        selected.append((trajectory, "top_ranked"))
        used_ids.add(trajectory.trajectory_id)

    remaining = [item for item in reversed(ranked) if item.trajectory_id not in used_ids]
    for trajectory in remaining[:low_count]:
        selected.append((trajectory, "low_ranked"))
        used_ids.add(trajectory.trajectory_id)

    random_pool = [item for item in ranked if item.trajectory_id not in used_ids]
    for trajectory in rng.sample(random_pool, min(random_count, len(random_pool))):
        selected.append((trajectory, "random"))
        used_ids.add(trajectory.trajectory_id)

    return selected


def build_lookahead_record(
    *,
    patient: dict[str, Any],
    mode: str,
    turn_number: int,
    openness_level: int,
    stage: str,
    base_conversation: list[Turn],
    trajectory: SearchTrajectory,
    selection_category: str,
    reward: dict[str, Any],
    selected: bool,
    max_context_pairs: int,
    reward_model_context_pairs: int,
    lookahead_depth: int,
    gamma: float,
) -> dict[str, Any]:
    root_step = trajectory.steps[0]
    root_candidate_conversation = [
        *base_conversation,
        Turn("Therapist", root_step["response"]),
    ]
    return {
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "mode": mode,
        "turn": turn_number,
        "dataset_stage": f"D{lookahead_depth}",
        "lookahead_depth": lookahead_depth,
        "discount_factor": gamma,
        "openness_level": openness_level,
        "stage": stage,
        "trajectory_id": trajectory.trajectory_id,
        "selection_category": selection_category,
        "candidate_id": root_step["candidate_id"],
        "candidate_response": root_step["response"],
        "candidate_metadata": root_step["candidate_metadata"],
        "branch_client_response": root_step["client_response"],
        "trajectory_steps": trajectory.steps,
        "predicted_cumulative_reward": trajectory.cumulative_reward,
        "reward_scores": reward["scores"],
        "final_score": reward["final_score"],
        "selected_for_continuation": selected,
        "reward_judge_raw": reward["raw"],
        "reward_model_input": format_reward_model_input(
            root_candidate_conversation,
            reward_model_context_pairs,
        ),
        "terminal_reward_model_input": format_reward_model_input(
            trajectory.conversation[:-1],
            reward_model_context_pairs,
        ),
        "judged_conversation": format_paired_history(
            trajectory.conversation,
            max_context_pairs,
        ),
    }


def generate_deep_lookahead_records(
    *,
    patient: dict[str, Any],
    mode: str,
    turn_number: int,
    openness_level: int,
    stage: str,
    base_conversation: list[Turn],
    therapist: AIMTherapist,
    client: SimulatedClient,
    reward_judge: RewardJudge,
    reward_scorer: RewardModelScorer,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_cbt_techniques = list(therapist.previous_cbt_techniques)
    candidates, metadata = therapist.generate_candidates(
        patient,
        base_conversation,
        openness_level,
    )
    root_cbt_techniques = list(therapist.previous_cbt_techniques)
    root_texts = [
        format_reward_model_input(
            [*base_conversation, Turn("Therapist", candidate["response"])],
            args.reward_model_context_pairs,
        )
        for candidate in candidates
    ]
    root_scores = reward_scorer.score(root_texts)
    trajectories = []
    for candidate, predicted_reward in zip(candidates, root_scores, strict=True):
        step = {
            "depth": 1,
            "candidate_id": candidate["id"],
            "response": candidate["response"],
            "candidate_metadata": metadata.get(candidate["id"]),
            "predicted_reward": predicted_reward,
            "client_response": None,
        }
        trajectories.append(
            SearchTrajectory(
                trajectory_id=candidate["id"],
                conversation=[
                    *base_conversation,
                    Turn("Therapist", candidate["response"]),
                ],
                steps=[step],
                cumulative_reward=predicted_reward,
                cbt_techniques=list(root_cbt_techniques),
            )
        )

    search_trace: dict[str, Any] = {
        "lookahead_depth": args.lookahead_depth,
        "beam_width": args.beam_width,
        "discount_factor": args.discount_factor,
        "depths": [
            {
                "depth": 1,
                "generated": len(trajectories),
                "trajectories": [
                    {
                        "trajectory_id": item.trajectory_id,
                        "predicted_cumulative_reward": item.cumulative_reward,
                    }
                    for item in trajectories
                ],
            }
        ],
    }

    for depth in range(2, args.lookahead_depth + 1):
        beam = sorted(trajectories, key=trajectory_sort_key, reverse=True)[
            : args.beam_width
        ]
        expanded: list[SearchTrajectory] = []
        for parent in beam:
            complete_latest_client_response(parent, client, openness_level)
            therapist.previous_cbt_techniques = list(parent.cbt_techniques)
            child_candidates, child_metadata = therapist.generate_candidates(
                patient,
                parent.conversation,
                openness_level,
            )
            child_cbt_techniques = list(therapist.previous_cbt_techniques)
            child_texts = [
                format_reward_model_input(
                    [*parent.conversation, Turn("Therapist", candidate["response"])],
                    args.reward_model_context_pairs,
                )
                for candidate in child_candidates
            ]
            child_scores = reward_scorer.score(child_texts)
            for candidate, predicted_reward in zip(
                child_candidates,
                child_scores,
                strict=True,
            ):
                step = {
                    "depth": depth,
                    "candidate_id": candidate["id"],
                    "response": candidate["response"],
                    "candidate_metadata": child_metadata.get(candidate["id"]),
                    "predicted_reward": predicted_reward,
                    "client_response": None,
                }
                steps = [*parent.steps, step]
                expanded.append(
                    SearchTrajectory(
                        trajectory_id=f"{parent.trajectory_id}.{candidate['id']}",
                        conversation=[
                            *parent.conversation,
                            Turn("Therapist", candidate["response"]),
                        ],
                        steps=steps,
                        cumulative_reward=cumulative_reward(
                            steps,
                            args.discount_factor,
                        ),
                        cbt_techniques=list(child_cbt_techniques),
                    )
                )
        trajectories = expanded
        search_trace["depths"].append(
            {
                "depth": depth,
                "expanded_from": [item.trajectory_id for item in beam],
                "generated": len(trajectories),
                "trajectories": [
                    {
                        "trajectory_id": item.trajectory_id,
                        "predicted_cumulative_reward": item.cumulative_reward,
                    }
                    for item in trajectories
                ],
            }
        )

    terminal = select_terminal_trajectories(
        trajectories,
        top_count=args.beam_width,
        random_count=args.random_terminal_candidates,
        low_count=args.low_ranked_terminal_candidates,
        rng=rng,
    )
    judged: list[tuple[SearchTrajectory, str, dict[str, Any]]] = []
    for trajectory, category in terminal:
        complete_latest_client_response(trajectory, client, openness_level)
        judged.append((trajectory, category, reward_judge.judge(trajectory.conversation)))

    best_trajectory, _, best_reward = max(
        judged,
        key=lambda item: (
            item[2]["final_score"],
            item[2]["scores"]["therapeutic_alliance"],
            item[2]["scores"]["motivational_interviewing_readiness"],
            item[0].cumulative_reward,
        ),
    )
    records = [
        build_lookahead_record(
            patient=patient,
            mode=mode,
            turn_number=turn_number,
            openness_level=openness_level,
            stage=stage,
            base_conversation=base_conversation,
            trajectory=trajectory,
            selection_category=category,
            reward=reward,
            selected=trajectory.trajectory_id == best_trajectory.trajectory_id,
            max_context_pairs=args.max_context_pairs,
            reward_model_context_pairs=args.reward_model_context_pairs,
            lookahead_depth=args.lookahead_depth,
            gamma=args.discount_factor,
        )
        for trajectory, category, reward in judged
    ]
    search_trace["terminal"] = [
        {
            "trajectory_id": trajectory.trajectory_id,
            "selection_category": category,
            "predicted_cumulative_reward": trajectory.cumulative_reward,
            "final_score": reward["final_score"],
        }
        for trajectory, category, reward in judged
    ]
    search_trace["selected_trajectory_id"] = best_trajectory.trajectory_id
    search_trace["selected_final_score"] = best_reward["final_score"]

    # Future steps are simulations only. Preserve state from root generation and
    # execute the first therapist-client pair of the best trajectory.
    therapist.previous_cbt_techniques = root_cbt_techniques or initial_cbt_techniques
    return records, search_trace


def simulate_reward_conversation(
    *,
    patient: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    openai_client: Any,
    model: str,
    openness_judge_model: str,
    reward_judge_model: str,
    reward_scorer: RewardModelScorer | None,
    rng: random.Random,
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
        args.max_context_pairs,
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
        elif args.lookahead_depth == 1:
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
                    max_context_pairs=args.max_context_pairs,
                    reward_model_context_pairs=args.reward_model_context_pairs,
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
        else:
            if reward_scorer is None:
                raise RuntimeError("Deep lookahead requires a reward-model scorer.")
            stage = stage_for_openness(openness_level_before_turn)
            records, search_trace = generate_deep_lookahead_records(
                patient=patient,
                mode=mode,
                turn_number=turn_number,
                openness_level=openness_level_before_turn,
                stage=stage,
                base_conversation=conversation,
                therapist=therapist,
                client=client,
                reward_judge=reward_judge,
                reward_scorer=reward_scorer,
                args=args,
                rng=rng,
            )
            best_record = next(
                record for record in records if record["selected_for_continuation"]
            )
            first_step = best_record["trajectory_steps"][0]
            conversation.append(Turn("Therapist", first_step["response"]))
            conversation.append(Turn("Client", first_step["client_response"]))
            training_records.extend(records)
            turn_traces.append(
                {
                    "turn": turn_number,
                    "openness_level": openness_level_before_turn,
                    "stage": stage,
                    "search": search_trace,
                    "candidates": records,
                    "selected_trajectory_id": best_record["trajectory_id"],
                    "selected_final_score": best_record["final_score"],
                    "therapist": first_step["response"],
                    "client": first_step["client_response"],
                }
            )

            if args.print:
                print(
                    f"{patient.get('id')} {mode} turn {turn_number}: "
                    f"{best_record['trajectory_id']} depth={args.lookahead_depth} "
                    f"score={best_record['final_score']:.2f}"
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
        "reward_model": str(args.reward_model) if args.reward_model else None,
        "lookahead_depth": args.lookahead_depth,
        "beam_width": args.beam_width,
        "discount_factor": args.discount_factor,
        "reward_model_context_pairs": args.reward_model_context_pairs,
        "max_context_pairs": args.max_context_pairs,
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


def trace_path_for_session(
    trace_dir: Path,
    patient_id: Any,
    mode: str,
    lookahead_depth: int,
) -> Path:
    filename = (
        f"session_{safe_patient_id(patient_id)}_{mode}_aim_reward.json"
        if lookahead_depth == 1
        else (
            f"session_{safe_patient_id(patient_id)}_{mode}_aim_reward"
            f"_d{lookahead_depth}.json"
        )
    )
    return trace_dir / mode / filename


def main() -> None:
    args = parse_args()
    if args.turns < 1:
        raise SystemExit("--turns must be at least 1.")
    if args.max_context_pairs < 0:
        raise SystemExit("--max-context-pairs must be 0 or greater.")
    if args.reward_model_context_pairs < 0:
        raise SystemExit("--reward-model-context-pairs must be 0 or greater.")
    if args.reward_model_max_length < 1:
        raise SystemExit("--reward-model-max-length must be at least 1.")
    if args.beam_width < 1:
        raise SystemExit("--beam-width must be at least 1.")
    if not 0.0 <= args.discount_factor <= 1.0:
        raise SystemExit("--discount-factor must be between 0 and 1.")
    if args.random_terminal_candidates < 0:
        raise SystemExit("--random-terminal-candidates must be 0 or greater.")
    if args.low_ranked_terminal_candidates < 0:
        raise SystemExit("--low-ranked-terminal-candidates must be 0 or greater.")
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume cannot be used together.")
    if args.lookahead_depth > 1 and args.reward_model is None:
        raise SystemExit("--reward-model is required when --lookahead-depth is 2 or 3.")
    if args.reward_model is not None and not args.reward_model.exists():
        raise SystemExit(f"Reward-model checkpoint does not exist: {args.reward_model}")

    load_environment()
    rng = random.Random(args.seed)
    model = args.model or openai_model()
    openness_judge_model = args.openness_judge_model or model
    reward_judge_model = args.reward_judge_model
    patients = load_patients_for_args(args, rng)
    openai_client = create_openai_client()
    reward_scorer = None
    if args.lookahead_depth > 1:
        reward_scorer = RewardModelScorer(
            args.reward_model,
            args.reward_model_max_length,
            args.reward_model_device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_filename = (
        "reward_training_data.jsonl"
        if args.lookahead_depth == 1
        else f"reward_training_data_d{args.lookahead_depth}.jsonl"
    )
    jsonl_path = args.output_file or args.output_dir / default_filename
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"

    if jsonl_path.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"Output exists. Use --overwrite to replace or --resume to continue: "
            f"{jsonl_path}"
        )
    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    total_sessions = len(patients) * len(args.modes)
    completed_sessions = 0
    skipped_sessions = 0
    total_records = 0

    for patient in patients:
        patient_id = patient.get("id")
        for mode in args.modes:
            completed_sessions += 1
            trace_path = trace_path_for_session(
                trace_dir,
                patient_id,
                mode,
                args.lookahead_depth,
            )
            if args.resume and trace_path.exists():
                skipped_sessions += 1
                print(
                    f"[skip] {completed_sessions}/{total_sessions} "
                    f"patient={patient_id} mode={mode} trace={trace_path}",
                    flush=True,
                )
                continue

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
                reward_scorer=reward_scorer,
                rng=rng,
            )
            records = trace["training_records"]
            append_jsonl(jsonl_path, records)
            total_records += len(records)

            write_trace(trace_path, trace)

    print(
        f"Saved {total_records} new reward records to {jsonl_path} "
        f"(skipped_sessions={skipped_sessions})"
    )
    print(f"Saved conversation traces to {trace_dir}")


if __name__ == "__main__":
    main()
