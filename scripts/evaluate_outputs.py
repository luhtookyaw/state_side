"""Evaluate generated conversations in outputs/.

This script reads conversation JSON files produced by scripts/simulate_conversation.py,
formats each transcript, evaluates it with the evaluators in src/, and writes
per-conversation evaluation JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

from src.alliance import evaluate_alliance  # noqa: E402
from src.ctrs import evaluate_ctrs_transcript  # noqa: E402
from src.miti import evaluate_miti_transcript  # noqa: E402
from src.therapist_skills import evaluate_therapist_skills  # noqa: E402
from src.wai_o import evaluate_wai_o_transcript  # noqa: E402
from src.wai_o_s import evaluate_wai_o_s_transcript  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated conversation JSON files in outputs/."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=(
            "Directory containing conversation JSON files. Searches recursively. "
            "Defaults to outputs/."
        ),
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
        default=os.getenv("EVALUATION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o"),
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
        "--sample-clients",
        type=int,
        help=(
            "Randomly select this many patient_id values, then evaluate all "
            "matching mode files for those clients."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used with --sample-clients.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversations whose evaluation JSON already exists.",
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
    parser.add_argument(
        "--skip-ctrs",
        action="store_true",
        help="Skip CTRS evaluation.",
    )
    parser.add_argument(
        "--skip-miti",
        action="store_true",
        help="Skip MITI evaluation.",
    )
    parser.add_argument(
        "--skip-wai-o",
        action="store_true",
        help="Skip WAI-O evaluation.",
    )
    parser.add_argument(
        "--skip-wai-o-s",
        action="store_true",
        help="Skip WAI-O-S evaluation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_id_for_file(path: Path) -> str | None:
    try:
        patient_id = load_json(path).get("patient_id")
    except (OSError, json.JSONDecodeError):
        return None
    return str(patient_id) if patient_id is not None else None


def sample_client_files(
    files: list[Path], sample_clients: int | None, seed: int | None
) -> list[Path]:
    if sample_clients is None:
        return files
    if sample_clients < 1:
        raise SystemExit("--sample-clients must be at least 1.")

    files_by_patient: dict[str, list[Path]] = {}
    for path in files:
        patient_id = patient_id_for_file(path)
        if patient_id is None:
            continue
        files_by_patient.setdefault(patient_id, []).append(path)

    patient_ids = sorted(files_by_patient)
    if not patient_ids:
        raise SystemExit("No patient_id values found in output JSON files.")

    rng = random.Random(seed)
    selected_ids = set(rng.sample(patient_ids, min(sample_clients, len(patient_ids))))
    selected_files = [
        path
        for patient_id in patient_ids
        if patient_id in selected_ids
        for path in sorted(files_by_patient[patient_id])
    ]
    return sorted(selected_files)


def conversation_files(
    outputs_dir: Path,
    max_files: int | None,
    sample_clients: int | None,
    seed: int | None,
) -> list[Path]:
    files = sorted(path for path in outputs_dir.rglob("*.json") if path.is_file())
    files = sample_client_files(files, sample_clients, seed)
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


def evaluate_conversation(
    openai_client: Any,
    model: str,
    temperature: float,
    path: Path,
    run_alliance: bool,
    run_therapist_skills: bool,
    run_ctrs: bool,
    run_miti: bool,
    run_wai_o: bool,
    run_wai_o_s: bool,
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

    def model_call(prompt: str, max_tokens: int) -> str:
        return call_model(openai_client, model, prompt, temperature, max_tokens=max_tokens)

    if run_alliance:
        result["evaluations"]["alliance"] = evaluate_alliance(transcript, model_call)

    if run_therapist_skills:
        result["evaluations"]["therapist_skills"] = evaluate_therapist_skills(
            transcript, model_call
        )

    if run_ctrs:
        result["evaluations"]["ctrs"] = evaluate_ctrs_transcript(transcript, model_call)

    if run_miti:
        result["evaluations"]["miti"] = evaluate_miti_transcript(transcript, model_call)

    if run_wai_o:
        result["evaluations"]["wai_o"] = evaluate_wai_o_transcript(transcript, model_call)

    if run_wai_o_s:
        result["evaluations"]["wai_o_s"] = evaluate_wai_o_s_transcript(
            transcript, model_call
        )

    return result


def evaluation_output_path(
    evaluations_dir: Path, outputs_dir: Path, conversation_path: Path
) -> Path:
    try:
        relative_path = conversation_path.relative_to(outputs_dir)
    except ValueError:
        relative_path = Path(conversation_path.name)

    return evaluations_dir / relative_path.parent / f"{relative_path.stem}_evaluation.json"


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

    files = conversation_files(
        args.outputs_dir,
        args.max_files,
        args.sample_clients,
        args.seed,
    )
    if not files:
        raise SystemExit(f"No JSON files found in {args.outputs_dir}.")

    openai_client = OpenAI()
    results = []
    args.evaluations_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, start=1):
        output_path = evaluation_output_path(args.evaluations_dir, args.outputs_dir, path)
        if args.skip_existing and output_path.exists():
            print(f"[{index}/{len(files)}] Skipping existing {output_path}")
            continue

        print(f"[{index}/{len(files)}] Evaluating {path}")
        result = evaluate_conversation(
            openai_client=openai_client,
            model=args.model,
            temperature=args.temperature,
            path=path,
            run_alliance=not args.skip_alliance,
            run_therapist_skills=not args.skip_therapist_skills,
            run_ctrs=not args.skip_ctrs,
            run_miti=not args.skip_miti,
            run_wai_o=not args.skip_wai_o,
            run_wai_o_s=not args.skip_wai_o_s,
        )
        results.append(result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
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
