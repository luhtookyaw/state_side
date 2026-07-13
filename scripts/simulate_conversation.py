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
from camel_therapist import (  # noqa: E402
    DEFAULT_CACTUS_CASES,
    DEFAULT_CAMEL_MODEL_ID,
    DEFAULT_CAMEL_VLLM_SERVER,
    CamelTherapist,
)
from flash_therapist import DEFAULT_FLASH_API_URL, FlashTherapist  # noqa: E402
from aim_therapist import AIMTherapist  # noqa: E402
from therapist import DEFAULT_THERAPIST_PROMPT, StandardTherapist  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
THERAPIST_TYPES = ("standard", "flash", "aim", "camel")


def clamp_openness_transition(raw_level: int, current_level: int) -> int:
    next_level = max(current_level - 1, min(current_level + 1, raw_level))
    return max(1, min(5, next_level))


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
        "--therapist-type",
        choices=THERAPIST_TYPES,
        default="standard",
        help="Therapist implementation to use. Defaults to standard.",
    )
    parser.add_argument(
        "--flash-api-url",
        default=os.getenv("FLASH_API_URL", DEFAULT_FLASH_API_URL),
        help=(
            "Base URL for the flash therapist API. Defaults to FLASH_API_URL, "
            "then http://localhost:8000."
        ),
    )
    parser.add_argument(
        "--camel-vllm-server",
        default=os.getenv("CAMEL_VLLM_SERVER", DEFAULT_CAMEL_VLLM_SERVER),
        help=(
            "OpenAI-compatible vLLM base URL for CAMEL. Defaults to "
            "CAMEL_VLLM_SERVER, then http://127.0.0.1:8000/v1."
        ),
    )
    parser.add_argument(
        "--camel-model-id",
        default=os.getenv("CAMEL_MODEL_ID", DEFAULT_CAMEL_MODEL_ID),
        help=(
            "CAMEL model id served by vLLM. Defaults to CAMEL_MODEL_ID, then "
            "LangAGI-Lab/camel."
        ),
    )
    parser.add_argument(
        "--camel-max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for CAMEL counselor responses. Defaults to 512.",
    )
    parser.add_argument(
        "--cactus-cases",
        type=Path,
        default=DEFAULT_CACTUS_CASES,
        help="Path to CACTUS case mapping JSON for CAMEL.",
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
        help="Path to standard therapist prompt template.",
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


def therapist_type_suffix(therapist_type: str) -> str:
    return "" if therapist_type == "standard" else f"_{therapist_type}"


def print_turn(
    turn_number: int,
    therapist: Turn,
    client: Turn,
    openness_level: int,
    openness_judgment: dict[str, Any] | None,
    selected_strategy: str | None,
    flash_response: dict[str, Any] | None,
    aim_response: dict[str, Any] | None,
    camel_response: dict[str, Any] | None,
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
    if selected_strategy is None:
        print("Selected strategy: not recorded")
    else:
        print(f"Selected strategy: {selected_strategy}")
    if flash_response is not None:
        technique = flash_response.get("technique")
        if isinstance(technique, str) and technique.strip():
            print(f"Technique: {technique.strip()}")
        else:
            print("Technique: not recorded")
    if aim_response is not None:
        print(f"AIM stage: {aim_response.get('stage')}")
        print(f"AIM selected: {aim_response.get('selected_response_id')}")
        selected_metadata = aim_response.get("selected_metadata")
        if isinstance(selected_metadata, dict):
            print(f"AIM selected agent: {selected_metadata.get('agent')}")
            cbt_recommendation = selected_metadata.get("cbt_recommendation")
            if isinstance(cbt_recommendation, dict):
                technique = cbt_recommendation.get("recommended_cbt_technique")
                print(f"CBT technique: {technique or 'not recorded'}")
                plan = cbt_recommendation.get("plan")
                print(f"CBT plan: {plan or 'not recorded'}")
    if camel_response is not None:
        print(f"CAMEL case: {camel_response.get('cactus_case_id')}")
        print(f"CBT technique: {camel_response.get('cbt_technique') or 'not recorded'}")
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
    therapist_prompt = args.therapist_prompt or DEFAULT_THERAPIST_PROMPT

    if args.therapist_type == "flash":
        therapist_role = FlashTherapist(args.flash_api_url)
    elif args.therapist_type == "aim":
        therapist_role = AIMTherapist(
            openai_client,
            model,
            args.temperature,
        )
    elif args.therapist_type == "camel":
        therapist_role = CamelTherapist(
            vllm_server=args.camel_vllm_server,
            model_id=args.camel_model_id,
            temperature=args.temperature,
            max_tokens=args.camel_max_tokens,
            cactus_cases_path=args.cactus_cases,
        )
    else:
        therapist_role = StandardTherapist(
            openai_client,
            model,
            args.temperature,
            prompt_path=therapist_prompt,
        )

    conversation: list[Turn] = []
    paired_turns: list[dict[str, Any]] = []
    mode_settings = MODE_SETTINGS[args.mode]
    openness_level = INITIAL_OPENNESS_LEVEL
    openness_judge_interval = mode_settings["openness_judge_interval"]
    openness_judgments: list[dict[str, Any]] = []
    selected_strategies: list[dict[str, Any]] = []
    flash_responses: list[dict[str, Any]] = []
    aim_responses: list[dict[str, Any]] = []
    camel_responses: list[dict[str, Any]] = []

    for turn_number in range(1, args.turns + 1):
        openness_level_before_turn = openness_level
        selected_strategy: str | None = None
        flash_response: dict[str, Any] | None = None
        aim_response: dict[str, Any] | None = None
        camel_response: dict[str, Any] | None = None
        if turn_number == 1:
            therapist_turn = Turn("Therapist", therapist_role.opening(patient))
            if args.therapist_type == "flash":
                flash_response = therapist_role.last_response_json
            elif args.therapist_type == "aim":
                aim_response = therapist_role.last_response_json
            elif args.therapist_type == "camel":
                camel_response = therapist_role.last_response_json
        else:
            if args.therapist_type == "aim":
                text = therapist_role.reply(
                    patient,
                    conversation,
                    openness_level_before_turn,
                )
            else:
                text = therapist_role.reply(patient, conversation)
            therapist_turn = Turn("Therapist", text)
            if args.therapist_type == "flash":
                flash_response = therapist_role.last_response_json
            elif args.therapist_type == "aim":
                aim_response = therapist_role.last_response_json
            elif args.therapist_type == "camel":
                camel_response = therapist_role.last_response_json

        if flash_response is not None:
            flash_response = {
                "turn": turn_number,
                **flash_response,
            }
            flash_responses.append(flash_response)

        if aim_response is not None:
            aim_response = {
                "turn": turn_number,
                **aim_response,
            }
            aim_responses.append(aim_response)
            selected_metadata = aim_response.get("selected_metadata")
            if isinstance(selected_metadata, dict):
                selected_strategy = str(selected_metadata.get("agent", "aim"))
            else:
                selected_strategy = f"composer:{aim_response.get('stage', 'unknown')}"
            selected_strategies.append(
                {
                    "turn": turn_number,
                    "strategy": selected_strategy,
                    "stage": aim_response.get("stage"),
                    "selected_response_id": aim_response.get("selected_response_id"),
                    "selected_metadata": selected_metadata,
                    "cbt_recommendation": (
                        selected_metadata.get("cbt_recommendation")
                        if isinstance(selected_metadata, dict)
                        else None
                    ),
                    "candidate_metadata": aim_response.get("candidate_metadata"),
                }
            )
        if camel_response is not None:
            camel_response = {
                "turn": turn_number,
                **camel_response,
            }
            camel_responses.append(camel_response)
            selected_strategy = "camel"
            selected_strategies.append(
                {
                    "turn": turn_number,
                    "strategy": selected_strategy,
                    "cactus_case_id": camel_response.get("cactus_case_id"),
                    "cbt_technique": camel_response.get("cbt_technique"),
                    "cbt_plan": camel_response.get("cbt_plan"),
                }
            )

        conversation.append(therapist_turn)

        text = client_role.reply(conversation, openness_level)
        client_turn = Turn("Client", text)
        conversation.append(client_turn)

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

        paired_turns.append(
            {
                "turn": turn_number,
                "openness_level": openness_level_before_turn,
                "therapist": therapist_turn.text,
                "client": client_turn.text,
                "openness_judgment": openness_judgment,
                "flash_response": flash_response,
                "aim_response": aim_response,
                "camel_response": camel_response,
                "strategy_used": selected_strategy,
            }
        )
        if args.print:
            print_turn(
                turn_number,
                therapist_turn,
                client_turn,
                openness_level_before_turn,
                openness_judgment,
                selected_strategy,
                flash_response,
                aim_response,
                camel_response,
            )

    return {
        "mode": args.mode,
        "therapist_type": args.therapist_type,
        "initial_openness_level": INITIAL_OPENNESS_LEVEL,
        "final_openness_level": openness_level,
        "openness_judge_interval": openness_judge_interval,
        "model": model,
        "openness_judge_model": openness_judge_model,
        "patient_id": patient.get("id"),
        "patient_name": patient.get("name"),
        "client_type": client_role.client_type,
        "openness_judgments": openness_judgments,
        "selected_strategies": selected_strategies,
        "flash_responses": flash_responses,
        "aim_responses": aim_responses,
        "camel_responses": camel_responses,
        "turns": paired_turns,
    }


def write_output(result: dict[str, Any], output_path: Path | None) -> Path:
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        patient_id = str(result["patient_id"]).replace("/", "-")
        therapist_suffix = therapist_type_suffix(str(result.get("therapist_type")))
        filename = f"conversation_{patient_id}_{result['mode']}{therapist_suffix}.json"
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
