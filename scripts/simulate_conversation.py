"""Simulate a therapy conversation between a client and therapist.

The simulator uses the prompt templates in ``prompts/`` and patient scenarios
from ``data/Patient_Psi_CM_Dataset.json``. Client difficulty controls the
client always starts at openness level 1. Client difficulty controls how often
openness is re-rated by the openness judge:

- easy: every 2 turns
- normal: every 4 turns
- hard: every 6 turns
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from chat_runtime import (  # noqa: E402
    DEFAULT_DATASET,
    ROOT_DIR,
    Turn,
    choose_patient,
    create_openai_client,
    load_dataset,
    load_environment,
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
from therapist import DEFAULT_THERAPIST_PROMPT, StandardTherapist  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a client-therapist conversation from a patient scenario."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_SETTINGS),
        default="normal",
        help="Client difficulty mode. Defaults to normal.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=8,
        help="Number of therapist-client exchange turns to generate. Defaults to 8.",
    )
    parser.add_argument(
        "--patient-id",
        help="Use a specific patient scenario id from the dataset, e.g. 1-1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used when choosing a patient scenario and client style.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--openness-judge-model",
        help=(
            "OpenAI model name for openness judging. Defaults to "
            "OPENNESS_JUDGE_MODEL, then --model."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for both speakers. Defaults to 0.8.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to patient scenario JSON dataset.",
    )
    parser.add_argument(
        "--client-prompt",
        type=Path,
        default=DEFAULT_CLIENT_PROMPT,
        help="Path to client prompt template.",
    )
    parser.add_argument(
        "--therapist-prompt",
        type=Path,
        default=DEFAULT_THERAPIST_PROMPT,
        help="Path to therapist prompt template.",
    )
    parser.add_argument(
        "--openness-judge-prompt",
        type=Path,
        default=DEFAULT_OPENNESS_JUDGE_PROMPT,
        help="Path to openness judge prompt template.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated conversation JSON.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print each conversation turn as it is generated.",
    )
    return parser.parse_args()


def print_turn(
    turn_number: int,
    therapist: Turn,
    client: Turn,
    openness_level: int,
    openness_judgment: dict[str, Any] | None,
) -> None:
    print(f"Turn: {turn_number}")
    print(f"Openness level used: {openness_level}")
    if openness_judgment is None:
        print("Openness judge: not run")
    else:
        print(
            "Openness judge: ran "
            f"(new openness level: {openness_judgment['openness_level']})"
        )
    print(f"Therapist: {therapist.text}")
    print(f"Client: {client.text}")
    print()


def simulate_conversation(args: argparse.Namespace) -> dict[str, Any]:
    if args.turns < 1:
        raise SystemExit("--turns must be at least 1.")

    load_environment()

    rng = random.Random(args.seed)
    model = args.model or openai_model()
    openness_judge_model = args.openness_judge_model or os.getenv(
        "OPENNESS_JUDGE_MODEL", model
    )
    dataset = load_dataset(args.dataset)
    patient = choose_patient(dataset, args.patient_id, rng)
    openai_client = create_openai_client()
    client_role = SimulatedClient(
        patient,
        args.mode,
        openai_client,
        model,
        args.temperature,
        prompt_path=args.client_prompt,
    )
    openness_judge = OpennessJudge(
        openai_client,
        openness_judge_model,
        args.temperature,
        prompt_path=args.openness_judge_prompt,
    )
    therapist_role = StandardTherapist(
        openai_client,
        model,
        args.temperature,
        prompt_path=args.therapist_prompt,
    )

    conversation: list[Turn] = []
    paired_turns: list[dict[str, Any]] = []
    mode_settings = MODE_SETTINGS[args.mode]
    openness_level = INITIAL_OPENNESS_LEVEL
    openness_judge_interval = mode_settings["openness_judge_interval"]
    openness_judgments: list[dict[str, Any]] = []

    for turn_number in range(1, args.turns + 1):
        openness_level_before_turn = openness_level
        if turn_number == 1:
            therapist_turn = Turn("Therapist", therapist_role.opening(patient))
        else:
            text = therapist_role.reply(patient, conversation)
            therapist_turn = Turn("Therapist", text)

        conversation.append(therapist_turn)

        text = client_role.reply(conversation, openness_level)
        client_turn = Turn("Client", text)
        conversation.append(client_turn)

        openness_judgment: dict[str, Any] | None = None
        if turn_number % openness_judge_interval == 0:
            openness_judgment = openness_judge.judge(conversation, turn_number)
            openness_level = openness_judgment["openness_level"]
            openness_judgments.append(openness_judgment)

        paired_turns.append(
            {
                "turn": turn_number,
                "openness_level": openness_level_before_turn,
                "therapist": therapist_turn.text,
                "client": client_turn.text,
                "openness_judgment": openness_judgment,
            }
        )
        if args.print:
            print_turn(
                turn_number,
                therapist_turn,
                client_turn,
                openness_level_before_turn,
                openness_judgment,
            )

    return {
        "mode": args.mode,
        "initial_openness_level": INITIAL_OPENNESS_LEVEL,
        "final_openness_level": openness_level,
        "openness_judge_interval": openness_judge_interval,
        "model": model,
        "openness_judge_model": openness_judge_model,
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "client_type": client_role.client_type,
        "openness_judgments": openness_judgments,
        "turns": paired_turns,
    }


def write_output(result: dict[str, Any], output_path: Path | None) -> Path:
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        patient_id = str(result["patient_id"]).replace("/", "-")
        filename = f"conversation_{patient_id}_{result['mode']}.json"
        output_path = DEFAULT_OUTPUT_DIR / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    result = simulate_conversation(args)
    output_path = write_output(result, args.output)

    print(f"Saved conversation to {output_path}")


if __name__ == "__main__":
    main()
