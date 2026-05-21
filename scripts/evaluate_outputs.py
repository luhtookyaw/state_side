"""Evaluate generated conversations in outputs/.

This script reads conversation JSON files produced by src/simulate_conversation.py,
formats each transcript, evaluates it with the prompts in scripts/alliance.py and
scripts/therapist_skills.py, and writes one combined JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_EVALUATIONS_DIR = ROOT_DIR / "evaluations"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.alliance import C_ALLIANCE_SYSTEM_PROMPT  # noqa: E402
from scripts.therapist_skills import (  # noqa: E402
    CBT_SPECIFIC_FOCUS,
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_STRATEGY,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING,
)


THERAPIST_SKILL_PROMPTS = {
    "guided_discovery": CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    "focus": CBT_SPECIFIC_FOCUS,
    "strategy": CBT_SPECIFIC_STRATEGY,
    "understanding": GEN_UNDERSTANDING,
    "interpersonal": GEN_INTERPERSONAL,
    "collaboration": GEN_COLLABORATION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated conversation JSON files in outputs/."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory containing conversation JSON files. Defaults to outputs/.",
    )
    parser.add_argument(
        "--evaluations-dir",
        type=Path,
        default=DEFAULT_EVALUATIONS_DIR,
        help="Directory for per-conversation evaluation JSON files.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help="Optional path for a combined evaluation JSON report.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EVALUATION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model used for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for evaluators. Defaults to 0.0.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on number of conversation files to evaluate.",
    )
    parser.add_argument(
        "--skip-alliance",
        action="store_true",
        help="Skip the alliance.py evaluation prompt.",
    )
    parser.add_argument(
        "--skip-therapist-skills",
        action="store_true",
        help="Skip therapist_skills.py evaluation prompts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def conversation_files(outputs_dir: Path, max_files: int | None) -> list[Path]:
    files = sorted(path for path in outputs_dir.glob("*.json") if path.is_file())
    if max_files is not None:
        files = files[:max_files]
    return files


def format_transcript(conversation: dict[str, Any]) -> str:
    turns = conversation.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise ValueError("Conversation JSON does not contain a non-empty 'turns' list.")

    lines: list[str] = []
    for turn in turns:
        turn_number = turn.get("turn", "?")
        therapist = str(turn.get("therapist", "")).strip()
        client = str(turn.get("client", "")).strip()
        if therapist:
            lines.append(f"Turn {turn_number} Therapist: {therapist}")
        if client:
            lines.append(f"Turn {turn_number} Client: {client}")
    return "\n".join(lines)


def uses_max_completion_tokens(model: str) -> bool:
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


def parse_leading_score(text: str) -> int | None:
    match = re.search(r"^\s*(\d+)", text)
    return int(match.group(1)) if match else None


def parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def parse_alliance_response(text: str) -> dict[str, Any]:
    """Parse alliance output, which often arrives as many JSON objects."""
    parsed = parse_json_response(text)
    if isinstance(parsed, dict):
        question = next((key for key in parsed if re.fullmatch(r"Q\d+", key)), None)
        return {
            "questions": [parsed],
            "by_question": {question: parsed} if question else {},
            "raw": text,
        }
    if isinstance(parsed, list):
        return {
            "questions": parsed,
            "by_question": {
                question: item
                for item in parsed
                if isinstance(item, dict)
                for question in item
                if re.fullmatch(r"Q\d+", question)
            },
            "raw": text,
        }

    questions: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            item, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(item, dict):
            questions.append(item)
        index = start + end

    by_question = {}
    for item in questions:
        question = next((key for key in item if re.fullmatch(r"Q\d+", key)), None)
        if question:
            by_question[question] = item

    return {
        "questions": questions,
        "by_question": by_question,
        "raw": text,
    }


def evaluate_conversation(
    openai_client: Any,
    model: str,
    temperature: float,
    path: Path,
    run_alliance: bool,
    run_therapist_skills: bool,
) -> dict[str, Any]:
    conversation = load_json(path)
    transcript = format_transcript(conversation)

    result: dict[str, Any] = {
        "file": str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path),
        "mode": conversation.get("mode"),
        "patient_id": conversation.get("patient_id"),
        "patient_name": conversation.get("patient_name"),
        "client_type": conversation.get("client_type"),
        "turn_count": len(conversation.get("turns", [])),
        "evaluations": {},
    }

    if run_alliance:
        prompt = C_ALLIANCE_SYSTEM_PROMPT.format(
            conversation=transcript,
            example=json.dumps(
                {
                    "Q1": "question or criterion text",
                    "score": "score",
                    "reason": "brief evidence-based reason",
                },
                ensure_ascii=False,
            ),
        )
        raw = call_model(openai_client, model, prompt, temperature, max_tokens=1400)
        result["evaluations"]["alliance"] = parse_alliance_response(raw)

    if run_therapist_skills:
        skills: dict[str, Any] = {}
        for name, template in THERAPIST_SKILL_PROMPTS.items():
            raw = call_model(
                openai_client,
                model,
                template.format(conversation=transcript),
                temperature,
                max_tokens=700,
            )
            skills[name] = {
                "score": parse_leading_score(raw),
                "raw": raw,
            }
        result["evaluations"]["therapist_skills"] = skills

    return result


def evaluation_output_path(evaluations_dir: Path, conversation_path: Path) -> Path:
    return evaluations_dir / f"{conversation_path.stem}_evaluation.json"


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install dependencies with "
            "`pip install -r requirements.txt` or activate the project virtualenv."
        ) from exc

    files = conversation_files(args.outputs_dir, args.max_files)
    if not files:
        raise SystemExit(f"No JSON files found in {args.outputs_dir}.")

    openai_client = OpenAI()
    results = []
    args.evaluations_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Evaluating {path}")
        result = evaluate_conversation(
            openai_client=openai_client,
            model=args.model,
            temperature=args.temperature,
            path=path,
            run_alliance=not args.skip_alliance,
            run_therapist_skills=not args.skip_therapist_skills,
        )
        results.append(result)

        output_path = evaluation_output_path(args.evaluations_dir, path)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {output_path}")

    if args.combined_output:
        report = {
            "model": args.model,
            "outputs_dir": str(args.outputs_dir),
            "count": len(results),
            "results": results,
        }
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        args.combined_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved combined evaluations to {args.combined_output}")


if __name__ == "__main__":
    main()
