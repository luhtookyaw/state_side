"""Generate conversations for every patient across all difficulty modes."""

from __future__ import annotations

import argparse
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
THERAPIST_TYPES = ("standard", "adaptive")


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
        "--readiness-judge-prompt",
        type=Path,
        help="Optional readiness judge prompt path for adaptive therapist runs.",
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        help="Optional limit on number of patients from the dataset.",
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
    therapist_suffix = "_adaptive" if therapist_type == "adaptive" else ""
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
    if args.readiness_judge_prompt:
        command.extend(["--readiness-judge-prompt", str(args.readiness_judge_prompt)])
    if args.print_turns:
        command.append("--print")
    return command


def main() -> None:
    args = parse_args()
    patients = load_patients(args.dataset, args.max_clients)

    completed = 0
    skipped = 0
    failed = 0
    total = len(patients) * len(args.modes)

    for mode in args.modes:
        (args.outputs_dir / mode).mkdir(parents=True, exist_ok=True)

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

            print(f"[run] {completed + skipped + failed + 1}/{total} {label} -> {path}")
            try:
                subprocess.run(command, cwd=ROOT_DIR, check=True)
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"[fail] {label} exited with code {exc.returncode}")
                if not args.continue_on_error:
                    raise SystemExit(exc.returncode) from exc
            else:
                completed += 1

    print(
        f"Done. completed={completed}, skipped={skipped}, failed={failed}, total={total}"
    )


if __name__ == "__main__":
    main()
