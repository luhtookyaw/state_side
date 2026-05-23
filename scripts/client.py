"""Client-side helpers for conversation simulation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import re
from typing import Any

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

DEFAULT_CLIENT_PROMPT = ROOT_DIR / "prompts" / "client_response.txt"
DEFAULT_OPENNESS_JUDGE_PROMPT = ROOT_DIR / "prompts" / "judges" / "openness_judge.txt"


def join_value(value: Any) -> str:
    if value is None:
        return "Not specified."
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) if value else "Not specified."
    return str(value)


def format_history(turns: list[Any], max_turns: int = 5) -> str:
    if not turns:
        return "No previous turns."
    recent_turns = turns[-max_turns:]
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in recent_turns)


def format_dialogue(turns: list[Any]) -> str:
    if not turns:
        return "No previous turns."
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


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


def format_client_prompt(
    template: str,
    patient: dict[str, Any],
    client_type: str,
    openness_level: int,
    conversation: list[Any],
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


def format_openness_judge_prompt(template: str, conversation: list[Any]) -> str:
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


class SimulatedClient:
    """Client role that owns its prompt and response generation."""

    def __init__(
        self,
        patient: dict[str, Any],
        mode: str,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path = DEFAULT_CLIENT_PROMPT,
    ) -> None:
        self.patient = patient
        self.mode = mode
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)
        self.client_type = choose_client_type(patient, mode)

    def reply(self, conversation: list[Any], openness_level: int) -> str:
        prompt = format_client_prompt(
            self.template,
            self.patient,
            self.client_type,
            openness_level,
            conversation,
        )
        return call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=220,
        )


class OpennessJudge:
    """Openness judge that owns its prompt and model call."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path = DEFAULT_OPENNESS_JUDGE_PROMPT,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)

    def judge(self, conversation: list[Any], turn_number: int) -> dict[str, Any]:
        prompt = format_openness_judge_prompt(self.template, conversation)
        judge_text = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=360,
        )
        return {
            "turn": turn_number,
            "openness_level": parse_openness_level(judge_text),
            "judge_response": judge_text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat directly with the simulated client.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_SETTINGS),
        default="normal",
        help="Client difficulty mode. Defaults to normal.",
    )
    parser.add_argument(
        "--patient-id",
        help="Use a specific patient scenario id from the dataset, e.g. 1-1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used when choosing a patient scenario.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Defaults to 0.8.",
    )
    parser.add_argument(
        "--openness-level",
        type=int,
        choices=range(1, 6),
        default=INITIAL_OPENNESS_LEVEL,
        help="Fixed openness level used for client replies. Defaults to 1.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()

    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset)
    patient = choose_patient(dataset, args.patient_id, rng)
    client_type = choose_client_type(patient, args.mode)
    model = args.model or openai_model()
    openai_client = create_openai_client()
    client_role = SimulatedClient(
        patient,
        args.mode,
        openai_client,
        model,
        args.temperature,
        prompt_path=args.client_prompt,
    )
    conversation: list[Turn] = []

    print(
        f"Talking to client {join_value(patient.get('name'))} "
        f"(patient_id={patient.get('id')}, mode={args.mode}, type={client_type})."
    )
    print("Type therapist messages. Type 'quit' or 'exit' to stop.")
    while True:
        user_text = input("Therapist> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        conversation.append(Turn("Therapist", user_text))
        text = client_role.reply(conversation, args.openness_level)
        conversation.append(Turn("Client", text))
        print(f"Client> {text}")


if __name__ == "__main__":
    main()
