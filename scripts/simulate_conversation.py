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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "Patient_Psi_CM_Dataset.json"
DEFAULT_CLIENT_PROMPT = ROOT_DIR / "prompts" / "client_response.txt"
DEFAULT_THERAPIST_PROMPT = ROOT_DIR / "prompts" / "therapist_response.txt"
DEFAULT_OPENNESS_JUDGE_PROMPT = ROOT_DIR / "prompts" / "judges" / "openness_judge.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


MODE_SETTINGS = {
    "easy": {"openness_judge_interval": 2},
    "normal": {"openness_judge_interval": 4},
    "hard": {"openness_judge_interval": 6},
}

INITIAL_OPENNESS_LEVEL = 1

CLIENT_TYPES_BY_MODE = {
    "easy": ("plain", "verbose", "ingratiating"),
    "normal": ("plain", "verbose", "go off on tangents", "ingratiating"),
    "hard": ("guarded", "hostile", "go off on tangents"),
}

CLIENT_TYPE_DESCRIPTIONS = {
    "go off on tangents": "prone to going off on tangents",
}


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


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
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4.1-mini.",
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


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing dataset file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Dataset is not valid JSON: {path}") from exc

    if not isinstance(data, list) or not data:
        raise SystemExit(f"Dataset must contain a non-empty JSON list: {path}")
    return data


def choose_patient(
    dataset: list[dict[str, Any]], patient_id: str | None, rng: random.Random
) -> dict[str, Any]:
    if patient_id:
        for patient in dataset:
            if str(patient.get("id")) == patient_id:
                return patient
        raise SystemExit(f"No patient scenario found with id {patient_id!r}.")
    return rng.choice(dataset)


def join_client_types(client_types: list[str]) -> str:
    descriptions = [
        CLIENT_TYPE_DESCRIPTIONS.get(client_type, client_type)
        for client_type in client_types
    ]
    if len(descriptions) == 1:
        return descriptions[0]
    if len(descriptions) == 2:
        return " and ".join(descriptions)
    return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"


def choose_client_type(patient: dict[str, Any], mode: str) -> str:
    client_types = patient.get("type") or ["plain"]
    if not isinstance(client_types, list):
        return str(client_types)

    mode_client_types = CLIENT_TYPES_BY_MODE[mode]
    available_client_types = set(client_types)
    matching_client_types = [
        client_type
        for client_type in mode_client_types
        if client_type in available_client_types
    ]
    if matching_client_types:
        return join_client_types(matching_client_types)
    return join_client_types([str(client_type) for client_type in client_types])


def join_value(value: Any) -> str:
    if value is None:
        return "Not specified."
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) if value else "Not specified."
    return str(value)


def format_history(turns: list[Turn], max_turns: int = 5) -> str:
    if not turns:
        return "No previous turns."
    recent_turns = turns[-max_turns:]
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in recent_turns)


def format_dialogue(turns: list[Turn]) -> str:
    if not turns:
        return "No previous turns."
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def format_client_prompt(
    template: str,
    patient: dict[str, Any],
    client_type: str,
    openness_level: int,
    conversation: list[Turn],
) -> str:
    fields = {
        "name": join_value(patient.get("name")),
        "type": client_type,
        "situation": join_value(patient.get("situation")),
        "auto_thought": join_value(patient.get("auto_thought")),
        "emotion": join_value(patient.get("emotion")),
        "behavior": join_value(patient.get("behavior")),
        "history": join_value(patient.get("history")),
        "intermediate_belief": join_value(patient.get("intermediate_belief")),
        "helpless_belief": join_value(patient.get("helpless_belief")),
        "unlovable_belief": join_value(patient.get("unlovable_belief")),
        "worthless_belief": join_value(patient.get("worthless_belief")),
        "coping_strategies": join_value(patient.get("coping_strategies")),
        "openness_level": openness_level,
        "conversation_history": format_history(conversation),
    }
    return template.format(**fields)


def format_therapist_prompt(
    template: str, patient: dict[str, Any], conversation: list[Turn]
) -> str:
    return template.format(
        name=join_value(patient.get("name")),
        conversation_history=format_history(conversation),
    )


def format_openness_judge_prompt(template: str, conversation: list[Turn]) -> str:
    return template.format(dialogue_context=format_dialogue(conversation))


def parse_openness_level(text: str) -> int:
    patterns = (
        r"Rating\s*:\s*([1-5])",
        r"Numerical Rating\s*:\s*([1-5])",
        r"\b([1-5])\s*(?:/|out of)\s*5\b",
        r"\b([1-5])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse openness rating from judge response: {text!r}")


def uses_max_completion_tokens(model: str) -> bool:
    """Return True for chat models that reject the legacy max_tokens parameter."""
    normalized_model = model.lower()
    return normalized_model.startswith(("gpt-5", "o1", "o3", "o4"))


def unsupported_parameter_error(exc: Exception, parameter: str) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("param") == parameter:
            return True

    message = str(exc).lower()
    return "unsupported parameter" in message and parameter.lower() in message


def call_model(client: Any, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    token_parameter = (
        "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"
    )
    request[token_parameter] = max_tokens

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        if token_parameter != "max_tokens" or not unsupported_parameter_error(
            exc, "max_tokens"
        ):
            raise
        request.pop("max_tokens")
        request["max_completion_tokens"] = max_tokens
        response = client.chat.completions.create(**request)

    text = response.choices[0].message.content
    return (text or "").strip()


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

    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install dependencies with "
            "`pip install -r requirements.txt` or activate the project virtualenv."
        ) from exc

    rng = random.Random(args.seed)
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openness_judge_model = args.openness_judge_model or os.getenv(
        "OPENNESS_JUDGE_MODEL", model
    )
    dataset = load_dataset(args.dataset)
    patient = choose_patient(dataset, args.patient_id, rng)
    client_type = choose_client_type(patient, args.mode)
    client_prompt = load_text(args.client_prompt)
    therapist_prompt = load_text(args.therapist_prompt)
    openness_judge_prompt = load_text(args.openness_judge_prompt)
    openai_client = OpenAI()

    conversation: list[Turn] = []
    paired_turns: list[dict[str, Any]] = []
    mode_settings = MODE_SETTINGS[args.mode]
    openness_level = INITIAL_OPENNESS_LEVEL
    openness_judge_interval = mode_settings["openness_judge_interval"]
    openness_judgments: list[dict[str, Any]] = []

    for turn_number in range(1, args.turns + 1):
        openness_level_before_turn = openness_level
        if turn_number == 1:
            therapist_turn = Turn(
                "Therapist",
                (
                    f"Hi {join_value(patient.get('name'))}, I'm glad you're here. "
                    "What feels most important for us to talk about today?"
                ),
            )
        else:
            prompt = format_therapist_prompt(therapist_prompt, patient, conversation)
            text = call_model(
                openai_client, model, prompt, args.temperature, max_tokens=260
            )
            therapist_turn = Turn("Therapist", text)

        conversation.append(therapist_turn)

        prompt = format_client_prompt(
            client_prompt,
            patient,
            client_type,
            openness_level,
            conversation,
        )
        text = call_model(openai_client, model, prompt, args.temperature, max_tokens=220)
        client_turn = Turn("Client", text)
        conversation.append(client_turn)

        openness_judgment: dict[str, Any] | None = None
        if turn_number % openness_judge_interval == 0:
            prompt = format_openness_judge_prompt(openness_judge_prompt, conversation)
            judge_text = call_model(
                openai_client,
                openness_judge_model,
                prompt,
                args.temperature,
                max_tokens=360,
            )
            openness_level = parse_openness_level(judge_text)
            openness_judgment = {
                "turn": turn_number,
                "openness_level": openness_level,
                "judge_response": judge_text,
            }
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
        "client_type": client_type,
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
