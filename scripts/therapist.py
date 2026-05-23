"""Therapist-side helpers for conversation simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from chat_runtime import (  # noqa: E402
    ROOT_DIR,
    Turn,
    call_model,
    create_openai_client,
    load_environment,
    load_text,
    openai_model,
)


DEFAULT_THERAPIST_PROMPT = ROOT_DIR / "prompts" / "therapist_response.txt"


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


def opening_therapist_message(patient: dict[str, Any]) -> str:
    return (
        f"Hi {join_value(patient.get('name'))}, I'm glad you're here. "
        "What feels most important for us to talk about today?"
    )


def format_therapist_prompt(
    template: str, patient: dict[str, Any], conversation: list[Any]
) -> str:
    return template.format(
        name=join_value(patient.get("name")),
        conversation_history=format_history(conversation),
    )


class StandardTherapist:
    """Therapist role that owns its prompt and response generation."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path = DEFAULT_THERAPIST_PROMPT,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)

    def opening(self, patient: dict[str, Any]) -> str:
        return opening_therapist_message(patient)

    def reply(self, patient: dict[str, Any], conversation: list[Any]) -> str:
        prompt = format_therapist_prompt(self.template, patient, conversation)
        return call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=260,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat directly with the therapist.")
    parser.add_argument(
        "--patient-name",
        default="Client",
        help="Client name used by the therapist. Defaults to Client.",
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
        "--therapist-prompt",
        type=Path,
        default=DEFAULT_THERAPIST_PROMPT,
        help="Path to therapist prompt template.",
    )
    parser.add_argument(
        "--no-opening",
        action="store_true",
        help="Do not print the therapist's default opening message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()

    patient = {"name": args.patient_name}
    model = args.model or openai_model()
    openai_client = create_openai_client()
    therapist = StandardTherapist(
        openai_client,
        model,
        args.temperature,
        prompt_path=args.therapist_prompt,
    )
    conversation: list[Turn] = []

    print(f"Talking to therapist for client {join_value(patient.get('name'))}.")
    print("Type client messages. Type 'quit' or 'exit' to stop.")
    if not args.no_opening:
        opening = therapist.opening(patient)
        conversation.append(Turn("Therapist", opening))
        print(f"Therapist> {opening}")

    while True:
        user_text = input("Client> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        conversation.append(Turn("Client", user_text))
        text = therapist.reply(patient, conversation)
        conversation.append(Turn("Therapist", text))
        print(f"Therapist> {text}")


if __name__ == "__main__":
    main()
