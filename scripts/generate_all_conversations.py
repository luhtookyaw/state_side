"""Generate conversations for every patient across all difficulty modes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "Patient_Psi_CM_Dataset.json"
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"
SIMULATOR = ROOT_DIR / "scripts" / "simulate_conversation.py"
MODES = ("easy", "normal", "hard")
THERAPIST_TYPES = ("standard", "adaptive", "flash", "hybrid")


@dataclass(frozen=True)
class ConversationJob:
    index: int
    label: str
    path: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one conversation per patient for each selected difficulty "
            "mode, writing results into outputs/<mode>/."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to patient scenario JSON dataset.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Root output directory. Mode folders are created inside this directory.",
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
        "--model",
        help="OpenAI model name passed through to simulate_conversation.py.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature passed through to simulate_conversation.py.",
    )
    parser.add_argument(
        "--therapist-type",
        choices=THERAPIST_TYPES,
        default="standard",
        help="Therapist implementation passed through to simulate_conversation.py.",
    )
    parser.add_argument(
        "--therapist-prompt",
        type=Path,
        help="Optional therapist prompt path passed through to simulate_conversation.py.",
    )
    parser.add_argument(
        "--readiness-judge-model",
        help="Optional readiness judge model for adaptive therapist runs.",
    )
    parser.add_argument(
        "--cbt-technique-chooser-model",
        help="Optional CBT technique chooser model for hybrid therapist runs.",
    )
    parser.add_argument(
        "--readiness-judge-prompt",
        type=Path,
        help="Optional readiness judge prompt path for adaptive therapist runs.",
    )
    parser.add_argument(
        "--flash-api-url",
        help=(
            "Optional flash therapist API base URL passed through to "
            "simulate_conversation.py."
        ),
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        help="Optional limit on number of patients from the dataset.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Maximum number of conversations to generate in parallel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate conversations even when the output file already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue generating remaining conversations if one run fails.",
    )
    parser.add_argument(
        "--print-turns",
        action="store_true",
        help="Forward --print to simulate_conversation.py for each run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output files without running the simulator.",
    )
    return parser.parse_args()


def load_patients(path: Path, max_clients: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        patients = json.load(dataset_file)

    if not isinstance(patients, list) or not patients:
        raise SystemExit(f"Dataset must contain a non-empty JSON list: {path}")

    if max_clients is not None:
        patients = patients[:max_clients]
    return patients


def safe_patient_id(patient_id: Any) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", str(patient_id)).strip("_")
    return safe_id or "unknown"


def output_path(
    outputs_dir: Path, patient_id: Any, mode: str, therapist_type: str
) -> Path:
    therapist_suffix = "" if therapist_type == "standard" else f"_{therapist_type}"
    filename = f"session_{safe_patient_id(patient_id)}_{mode}{therapist_suffix}.json"
    return outputs_dir / mode / filename


def simulator_command(
    args: argparse.Namespace, patient_id: Any, mode: str, output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(SIMULATOR),
        "--mode",
        mode,
        "--therapist-type",
        args.therapist_type,
        "--turns",
        str(args.turns),
        "--patient-id",
        str(patient_id),
        "--dataset",
        str(args.dataset),
        "--temperature",
        str(args.temperature),
        "--output",
        str(output),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.therapist_prompt:
        command.extend(["--therapist-prompt", str(args.therapist_prompt)])
    if args.readiness_judge_model:
        command.extend(["--readiness-judge-model", args.readiness_judge_model])
    if args.cbt_technique_chooser_model:
        command.extend(
            ["--cbt-technique-chooser-model", args.cbt_technique_chooser_model]
        )
    if args.readiness_judge_prompt:
        command.extend(["--readiness-judge-prompt", str(args.readiness_judge_prompt)])
    if args.flash_api_url:
        command.extend(["--flash-api-url", args.flash_api_url])
    if args.print_turns:
        command.append("--print")
    return command


def run_job(job: ConversationJob, total: int) -> tuple[ConversationJob, int]:
    print(f"[run] {job.index}/{total} {job.label} -> {job.path}", flush=True)
    result = subprocess.run(job.command, cwd=ROOT_DIR, check=False)
    return job, result.returncode


def main() -> None:
    args = parse_args()
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be at least 1")

    patients = load_patients(args.dataset, args.max_clients)

    completed = 0
    skipped = 0
    failed = 0
    total = len(patients) * len(args.modes)

    for mode in args.modes:
        (args.outputs_dir / mode).mkdir(parents=True, exist_ok=True)

    jobs: list[ConversationJob] = []
    for patient in patients:
        patient_id = patient.get("id")
        if patient_id is None:
            raise SystemExit("Every patient record must include an 'id'.")

        for mode in args.modes:
            path = output_path(args.outputs_dir, patient_id, mode, args.therapist_type)
            label = f"{patient_id} {mode} {args.therapist_type}"

            if path.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip] {label} -> {path}")
                continue

            command = simulator_command(args, patient_id, mode, path)
            if args.dry_run:
                print(f"[dry-run] {label} -> {path}")
                continue

            jobs.append(
                ConversationJob(
                    index=completed + skipped + failed + len(jobs) + 1,
                    label=label,
                    path=path,
                    command=command,
                )
            )

    first_failure_code = 0
    if args.max_jobs == 1:
        for job in jobs:
            job, returncode = run_job(job, total)
            if returncode:
                failed += 1
                print(f"[fail] {job.label} exited with code {returncode}", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(returncode)
            else:
                completed += 1
                print(f"[done] {job.label} -> {job.path}", flush=True)

        print(
            f"Done. completed={completed}, skipped={skipped}, failed={failed}, "
            f"total={total}"
        )
        return

    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {
            executor.submit(run_job, job, total): job
            for job in jobs
        }

        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode:
                failed += 1
                first_failure_code = first_failure_code or returncode
                print(f"[fail] {job.label} exited with code {returncode}", flush=True)
            else:
                completed += 1
                print(f"[done] {job.label} -> {job.path}", flush=True)

    print(
        f"Done. completed={completed}, skipped={skipped}, failed={failed}, total={total}"
    )
    if first_failure_code and not args.continue_on_error:
        raise SystemExit(first_failure_code)


if __name__ == "__main__":
    main()
