"""Adaptive therapist helpers for readiness-based mode selection."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
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
from therapist import format_history, join_value, opening_therapist_message  # noqa: E402

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modes import CBT_MODE, MI_MODE, MI_SUPPORTED_CBT_MODE  # noqa: E402


DEFAULT_ADAPTIVE_THERAPIST_PROMPT = (
    ROOT_DIR / "prompts" / "adaptive_therapist_response.txt"
)
DEFAULT_READINESS_JUDGE_PROMPT = ROOT_DIR / "prompts" / "judges" / "readiness_judge.txt"

SENTIMENT_SCORES = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1,
}
READINESS_LEVEL_SCORES = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}
READINESS_VALUE_ALIASES = {
    "Sentiment": {
        "Low": "Negative",
        "Medium": "Neutral",
        "High": "Positive",
    },
    "Motivation": {
        "Negative": "Low",
        "Neutral": "Medium",
        "Positive": "High",
    },
    "Engagement": {
        "Negative": "Low",
        "Neutral": "Medium",
        "Positive": "High",
    },
}
MODE_BY_NAME = {
    "MI": MI_MODE,
    "MI-supported CBT": MI_SUPPORTED_CBT_MODE,
    "CBT": CBT_MODE,
}
RECENT_STRATEGY_LIMIT = 3


def format_readiness_judge_prompt(template: str, conversation: list[Any]) -> str:
    return (
        f"{template.rstrip()}\n\n"
        "Dialogue Context:\n"
        f"{format_history(conversation, max_turns=6)}"
    )


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

    expanded_candidates = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        normalized = re.sub(r"^\s*\{\{", "{", candidate)
        normalized = re.sub(r"\}\}\s*$", "}", normalized)
        expanded_candidates.append(normalized)
    return expanded_candidates


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


def normalized_choice(value: Any, allowed: set[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {value!r}")

    for allowed_value in allowed:
        if value.strip().lower() == allowed_value.lower():
            return allowed_value
    raise ValueError(
        f"{field_name} must be one of {sorted(allowed)}, got {value!r}"
    )


def normalized_readiness_choice(value: Any, allowed: set[str], field_name: str) -> str:
    try:
        return normalized_choice(value, allowed, field_name)
    except ValueError:
        if not isinstance(value, str):
            raise

        aliases = READINESS_VALUE_ALIASES.get(field_name, {})
        for alias, normalized_value in aliases.items():
            if value.strip().lower() == alias.lower():
                return normalized_value
        raise


def compute_readiness_score(judgment: dict[str, Any]) -> int:
    motivation = normalized_readiness_choice(
        judgment.get("Motivation"), set(READINESS_LEVEL_SCORES), "Motivation"
    )
    engagement = normalized_readiness_choice(
        judgment.get("Engagement"), set(READINESS_LEVEL_SCORES), "Engagement"
    )
    sentiment = normalized_readiness_choice(
        judgment.get("Sentiment"), set(SENTIMENT_SCORES), "Sentiment"
    )
    return (
        READINESS_LEVEL_SCORES[motivation] * 2
        + READINESS_LEVEL_SCORES[engagement]
        + SENTIMENT_SCORES[sentiment]
    )


def parse_readiness_judgment_text(text: str) -> dict[str, Any]:
    judgment: dict[str, Any] = {}
    fields = {
        "Sentiment": set(SENTIMENT_SCORES),
        "Motivation": set(READINESS_LEVEL_SCORES),
        "Engagement": set(READINESS_LEVEL_SCORES),
    }
    for field_name, allowed_values in fields.items():
        allowed_pattern = "|".join(sorted(allowed_values))
        pattern = rf'"?{field_name}"?\s*:\s*"?({allowed_pattern})"?'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Could not parse {field_name} from readiness judge output: {text!r}"
            )
        judgment[field_name] = normalized_choice(
            match.group(1),
            allowed_values,
            field_name,
        )
    return judgment


def readiness_mode_name(readiness_score: int) -> str:
    if readiness_score <= 4:
        return "MI"
    if readiness_score <= 7:
        return "MI-supported CBT"
    return "CBT"


def format_recent_strategies(
    strategies: list[str],
    max_strategies: int = RECENT_STRATEGY_LIMIT,
) -> str:
    recent = [strategy for strategy in strategies[-max_strategies:] if strategy.strip()]
    if not recent:
        return "None yet."
    return "\n".join(
        f"{index}. {strategy}" for index, strategy in enumerate(recent, start=1)
    )


def format_adaptive_therapist_prompt(
    template: str,
    patient: dict[str, Any],
    conversation: list[Any],
    mode: str,
    recent_strategies: list[str] | None = None,
) -> str:
    return template.format(
        name=join_value(patient.get("name")),
        mode=mode,
        recent_strategies=format_recent_strategies(recent_strategies or []),
        conversation_history=format_history(conversation),
    )


def parse_therapist_response(text: str) -> tuple[str, dict[str, Any]]:
    parsed = extract_json_object(text)
    response = parsed.get("therapist_response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Missing therapist_response in model output: {text!r}")
    return response.strip(), parsed


class ReadinessJudge:
    """Readiness judge that rates recent client utterances."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path = DEFAULT_READINESS_JUDGE_PROMPT,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)

    def judge(self, conversation: list[Any]) -> dict[str, Any]:
        prompt = format_readiness_judge_prompt(self.template, conversation)
        judge_text = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=180,
        )
        try:
            judgment = extract_json_object(judge_text)
        except ValueError:
            judgment = parse_readiness_judgment_text(judge_text)
        readiness_score = compute_readiness_score(judgment)
        mode_name = readiness_mode_name(readiness_score)
        return {
            "sentiment": normalized_readiness_choice(
                judgment.get("Sentiment"), set(SENTIMENT_SCORES), "Sentiment"
            ),
            "motivation": normalized_readiness_choice(
                judgment.get("Motivation"), set(READINESS_LEVEL_SCORES), "Motivation"
            ),
            "engagement": normalized_readiness_choice(
                judgment.get("Engagement"), set(READINESS_LEVEL_SCORES), "Engagement"
            ),
            "readiness_score": readiness_score,
            "mode": mode_name,
            "judge_response": judge_text,
        }


class AdaptiveTherapist:
    """Therapist role that adapts MI/CBT mode from readiness judgments."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        prompt_path: Path = DEFAULT_ADAPTIVE_THERAPIST_PROMPT,
        readiness_judge_model: str | None = None,
        readiness_judge_prompt_path: Path = DEFAULT_READINESS_JUDGE_PROMPT,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.template = load_text(prompt_path)
        self.readiness_judge = ReadinessJudge(
            openai_client,
            readiness_judge_model or model,
            temperature,
            prompt_path=readiness_judge_prompt_path,
        )
        self.last_readiness_judgment: dict[str, Any] | None = None
        self.last_response_json: dict[str, Any] | None = None
        self.recent_strategies: list[str] = []

    def opening(self, patient: dict[str, Any]) -> str:
        return opening_therapist_message(patient)

    def reply(self, patient: dict[str, Any], conversation: list[Any]) -> str:
        readiness_judgment = self.readiness_judge.judge(conversation)
        self.last_readiness_judgment = readiness_judgment
        mode = MODE_BY_NAME[readiness_judgment["mode"]]
        prompt = format_adaptive_therapist_prompt(
            self.template,
            patient,
            conversation,
            mode,
            self.recent_strategies,
        )
        response_text = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=320,
        )
        therapist_response, response_json = parse_therapist_response(response_text)
        self.last_response_json = response_json
        strategy_used = response_json.get("strategy_used")
        if isinstance(strategy_used, str) and strategy_used.strip():
            self.recent_strategies.append(strategy_used.strip())
        return therapist_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat directly with the adaptive readiness-based therapist."
    )
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
        "--readiness-judge-model",
        help="OpenAI model name for readiness judging. Defaults to --model.",
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
        default=DEFAULT_ADAPTIVE_THERAPIST_PROMPT,
        help="Path to adaptive therapist prompt template.",
    )
    parser.add_argument(
        "--readiness-judge-prompt",
        type=Path,
        default=DEFAULT_READINESS_JUDGE_PROMPT,
        help="Path to readiness judge prompt template.",
    )
    parser.add_argument(
        "--no-opening",
        action="store_true",
        help="Do not print the therapist's default opening message.",
    )
    parser.add_argument(
        "--show-readiness",
        action="store_true",
        help="Print readiness judgment metadata before each therapist response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()

    patient = {"name": args.patient_name}
    model = args.model or openai_model()
    openai_client = create_openai_client()
    therapist = AdaptiveTherapist(
        openai_client,
        model,
        args.temperature,
        prompt_path=args.therapist_prompt,
        readiness_judge_model=args.readiness_judge_model,
        readiness_judge_prompt_path=args.readiness_judge_prompt,
    )
    conversation: list[Turn] = []

    print(f"Talking to adaptive therapist for client {join_value(patient.get('name'))}.")
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

        if args.show_readiness and therapist.last_readiness_judgment is not None:
            readiness = therapist.last_readiness_judgment
            print(
                "Readiness> "
                f"score={readiness['readiness_score']}, mode={readiness['mode']}, "
                f"sentiment={readiness['sentiment']}, "
                f"motivation={readiness['motivation']}, "
                f"engagement={readiness['engagement']}"
            )
        print(f"Therapist> {text}")


if __name__ == "__main__":
    main()
