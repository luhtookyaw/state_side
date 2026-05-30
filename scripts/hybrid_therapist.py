"""Hybrid MI/CBT therapist helper."""

from __future__ import annotations

import argparse
import ast
import json
import re
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
from therapist import format_history, join_value  # noqa: E402


DEFAULT_HYBRID_PROMPT_DIR = ROOT_DIR / "prompts" / "hybrid_therapist"
DEFAULT_OPENING_THERAPIST_PROMPT = (
    DEFAULT_HYBRID_PROMPT_DIR / "opening_therapist_response.txt"
)
DEFAULT_MI_THERAPIST_PROMPT = (
    DEFAULT_HYBRID_PROMPT_DIR / "mi_therapist_response.txt"
)
DEFAULT_CBT_TECHNIQUE_CHOOSER_PROMPT = (
    DEFAULT_HYBRID_PROMPT_DIR / "cbt_technique_chooser.txt"
)
DEFAULT_CBT_THERAPIST_PROMPT = (
    DEFAULT_HYBRID_PROMPT_DIR / "cbt_therapist_response.txt"
)
CBT_OPENNESS_THRESHOLD = 3


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def json_candidates(text: str) -> list[str]:
    stripped = strip_code_fence(text)
    candidates = [stripped]

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    return candidates


def extract_json_object(text: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for candidate in json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, TypeError, ValueError) as literal_exc:
                last_error = literal_exc
                continue

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got: {type(parsed).__name__}")
        return parsed

    raise ValueError(f"Could not parse JSON object from model output: {text!r}") from last_error


def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def format_mi_therapist_prompt(
    template: str,
    patient: dict[str, Any],
    conversation: list[Any],
) -> str:
    return render_template(
        template,
        {
            "name": join_value(patient.get("name")),
            "conversation_history": format_history(conversation),
        },
    )


def format_opening_therapist_prompt(template: str, patient: dict[str, Any]) -> str:
    return render_template(
        template,
        {"name": join_value(patient.get("name"))},
    )


def format_cbt_technique_prompt(
    template: str,
    conversation: list[Any],
    previous_cbt_techniques: list[str],
) -> str:
    return render_template(
        template,
        {
            "conversation_history": format_history(conversation),
            "previous_cbt_techniques": format_previous_cbt_techniques(
                previous_cbt_techniques
            ),
        },
    )


def format_cbt_therapist_prompt(
    template: str,
    conversation: list[Any],
    cbt_recommendation: dict[str, Any],
) -> str:
    return render_template(
        template,
        {
            "conversation_history": format_history(conversation),
            "cbt_recommendation": format_cbt_recommendation(cbt_recommendation),
        },
    )


def format_previous_cbt_techniques(techniques: list[str]) -> str:
    if not techniques:
        return "None yet."
    return "\n".join(
        f"{index}. {technique}"
        for index, technique in enumerate(techniques, start=1)
    )


def format_cbt_recommendation(recommendation: dict[str, Any]) -> str:
    fields = [
        ("Recommended technique", recommendation.get("recommended_cbt_technique")),
        ("Goal", recommendation.get("goal")),
        ("Reasoning", recommendation.get("reasoning")),
    ]
    lines = []
    for label, value in fields:
        lines.append(f"{label}: {join_value(value)}")
    return "\n".join(lines)


class HybridTherapist:
    """Therapist role that switches from MI to CBT based on openness."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        opening_prompt_path: Path = DEFAULT_OPENING_THERAPIST_PROMPT,
        mi_prompt_path: Path = DEFAULT_MI_THERAPIST_PROMPT,
        cbt_technique_chooser_prompt_path: Path = DEFAULT_CBT_TECHNIQUE_CHOOSER_PROMPT,
        cbt_therapist_prompt_path: Path = DEFAULT_CBT_THERAPIST_PROMPT,
        cbt_technique_chooser_model: str | None = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.cbt_technique_chooser_model = cbt_technique_chooser_model or model
        self.opening_template = load_text(opening_prompt_path)
        self.mi_template = load_text(mi_prompt_path)
        self.cbt_technique_chooser_template = load_text(
            cbt_technique_chooser_prompt_path
        )
        self.cbt_therapist_template = load_text(cbt_therapist_prompt_path)
        self.previous_cbt_techniques: list[str] = []
        self.last_cbt_recommendation: dict[str, Any] | None = None
        self.last_mi_response: str | None = None
        self.last_cbt_response: str | None = None
        self.last_response_mode: str | None = None

    def _mi_reply(self, patient: dict[str, Any], conversation: list[Any]) -> str:
        prompt = format_mi_therapist_prompt(self.mi_template, patient, conversation)
        response = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=260,
        )
        self.last_mi_response = response
        self.last_response_mode = "MI"
        return response

    def _choose_cbt_technique(self, conversation: list[Any]) -> dict[str, Any]:
        prompt = format_cbt_technique_prompt(
            self.cbt_technique_chooser_template,
            conversation,
            self.previous_cbt_techniques,
        )
        response = call_model(
            self.openai_client,
            self.cbt_technique_chooser_model,
            prompt,
            self.temperature,
            max_tokens=360,
        )
        recommendation = extract_json_object(response)
        technique = recommendation.get("recommended_cbt_technique")
        if not isinstance(technique, str) or not technique.strip():
            raise ValueError(
                f"Missing recommended_cbt_technique in model output: {response!r}"
        )
        self.previous_cbt_techniques.append(technique.strip())
        self.last_cbt_recommendation = recommendation
        return recommendation

    def _cbt_reply(
        self,
        conversation: list[Any],
        cbt_recommendation: dict[str, Any],
    ) -> str:
        prompt = format_cbt_therapist_prompt(
            self.cbt_therapist_template,
            conversation,
            cbt_recommendation,
        )
        response = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=260,
        )
        self.last_cbt_response = response
        self.last_response_mode = "CBT"
        return response

    def opening(self, patient: dict[str, Any]) -> str:
        prompt = format_opening_therapist_prompt(self.opening_template, patient)
        return call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=160,
        )

    def reply(
        self,
        patient: dict[str, Any],
        conversation: list[Any],
        openness_level: int = 1,
    ) -> str:
        if openness_level <= CBT_OPENNESS_THRESHOLD:
            self.last_cbt_response = None
            return self._mi_reply(patient, conversation)

        cbt_recommendation = self._choose_cbt_technique(conversation)
        return self._cbt_reply(conversation, cbt_recommendation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat directly with the hybrid therapist.")
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
        "--cbt-technique-chooser-model",
        help="OpenAI model name for CBT technique selection. Defaults to --model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Defaults to 0.8.",
    )
    parser.add_argument(
        "--opening-prompt",
        type=Path,
        default=DEFAULT_OPENING_THERAPIST_PROMPT,
        help="Path to opening therapist prompt template.",
    )
    parser.add_argument(
        "--mi-prompt",
        type=Path,
        default=DEFAULT_MI_THERAPIST_PROMPT,
        help="Path to MI therapist prompt template.",
    )
    parser.add_argument(
        "--cbt-technique-chooser-prompt",
        type=Path,
        default=DEFAULT_CBT_TECHNIQUE_CHOOSER_PROMPT,
        help="Path to CBT technique chooser prompt template.",
    )
    parser.add_argument(
        "--cbt-therapist-prompt",
        type=Path,
        default=DEFAULT_CBT_THERAPIST_PROMPT,
        help="Path to CBT therapist prompt template.",
    )
    parser.add_argument(
        "--openness-level",
        type=int,
        default=1,
        help="Openness level used in direct chat mode. CBT is used when greater than 3.",
    )
    parser.add_argument(
        "--show-hybrid",
        action="store_true",
        help="Print hybrid routing and CBT recommendation metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()

    patient = {"name": args.patient_name}
    model = args.model or openai_model()
    openai_client = create_openai_client()
    therapist = HybridTherapist(
        openai_client,
        model,
        args.temperature,
        opening_prompt_path=args.opening_prompt,
        mi_prompt_path=args.mi_prompt,
        cbt_technique_chooser_prompt_path=args.cbt_technique_chooser_prompt,
        cbt_therapist_prompt_path=args.cbt_therapist_prompt,
        cbt_technique_chooser_model=args.cbt_technique_chooser_model,
    )
    conversation: list[Turn] = []

    print(f"Talking to hybrid therapist for client {join_value(patient.get('name'))}.")
    print("Type client messages. Type 'quit' or 'exit' to stop.")
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
        text = therapist.reply(patient, conversation, args.openness_level)
        conversation.append(Turn("Therapist", text))

        if args.show_hybrid:
            recommendation = therapist.last_cbt_recommendation
            if args.openness_level <= CBT_OPENNESS_THRESHOLD:
                print(f"Hybrid> MI route (openness={args.openness_level})")
            else:
                print(
                    "Hybrid> "
                    f"CBT route (openness={args.openness_level}), "
                    f"cbt_technique={recommendation.get('recommended_cbt_technique')}"
                )
        print(f"Therapist> {text}")


if __name__ == "__main__":
    main()
