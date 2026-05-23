"""Report aggregate evaluation results by difficulty mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATIONS_DIR = ROOT_DIR / "evaluations"
MODES = ("easy", "normal", "hard")
MODE_LABELS = {
    "easy": "Easy",
    "normal": "Normal",
    "hard": "Hard",
}
SKILL_COLUMNS = (
    ("guided_discovery", "Discovery"),
    ("focus", "Focus"),
    ("strategy", "Strategy"),
    ("understanding", "Understanding"),
    ("interpersonal", "Interpersonal"),
    ("collaboration", "Collaboration"),
)
METRIC_COLUMNS = (
    ("ctrs", "CTRS"),
    ("wai", "WAI"),
    ("miti", "MITI"),
    ("alliance", "Alli"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print average therapist skill tables from evaluation JSON files."
    )
    parser.add_argument(
        "--evaluations-dir",
        type=Path,
        default=DEFAULT_EVALUATIONS_DIR,
        help="Root evaluations directory containing easy/normal/hard folders.",
    )
    parser.add_argument(
        "--approach",
        default="gpt-4o-mini",
        help="Approach label to print in the table row. Defaults to gpt-4o-mini.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Modes to report. Defaults to easy normal hard.",
    )
    parser.add_argument(
        "--format",
        choices=("latex", "markdown"),
        default="latex",
        help="Output table format. Defaults to latex.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=("therapist-skills", "summary"),
        default=["therapist-skills", "summary"],
        help="Tables to print. Defaults to both therapist-skills and summary.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_value(evaluation: dict[str, Any], skill: str) -> float | None:
    value = (
        evaluation.get("evaluations", {})
        .get("therapist_skills", {})
        .get(skill, {})
        .get("score")
    )
    return as_float(value)


def mode_averages(evaluations_dir: Path, mode: str) -> tuple[dict[str, float], int]:
    files = sorted((evaluations_dir / mode).glob("*.json"))
    scores: dict[str, list[float]] = {skill: [] for skill, _ in SKILL_COLUMNS}

    for path in files:
        evaluation = load_json(path)
        for skill, _ in SKILL_COLUMNS:
            score = score_value(evaluation, skill)
            if score is not None:
                scores[skill].append(score)

    averages = {
        skill: mean(values)
        for skill, values in scores.items()
        if values
    }
    return averages, len(files)


def ctrs_score(evaluation: dict[str, Any]) -> float | None:
    ctrs = evaluation.get("evaluations", {}).get("ctrs", {})
    total_score = as_float(ctrs.get("total_score"))
    if total_score is not None:
        return total_score / 11

    item_scores: list[float] = []
    for section in ("part_I", "part_II"):
        for item in ctrs.get(section, {}).values():
            if isinstance(item, dict):
                score = as_float(item.get("score"))
                if score is not None:
                    item_scores.append(score)
    return mean(item_scores) if item_scores else None


def wai_score(evaluation: dict[str, Any]) -> float | None:
    evaluations = evaluation.get("evaluations", {})
    wai_o_s = evaluations.get("wai_o_s", {})
    score = as_float(wai_o_s.get("total_score"))
    if score is not None:
        return score

    observers = evaluations.get("wai_o", {}).get("observers", [])
    if observers:
        scores = observers[0].get("scores", {})
        total_score = as_float(scores.get("total_score"))
        if total_score is not None:
            return total_score / 36
    return None


def miti_score(evaluation: dict[str, Any]) -> float | None:
    global_ratings = (
        evaluation.get("evaluations", {})
        .get("miti", {})
        .get("global_ratings", {})
    )
    scores = [
        score
        for rating in global_ratings.values()
        if isinstance(rating, dict)
        for score in [as_float(rating.get("score"))]
        if score is not None
    ]
    return mean(scores) if scores else None


def alliance_score(evaluation: dict[str, Any]) -> float | None:
    by_question = (
        evaluation.get("evaluations", {})
        .get("alliance", {})
        .get("by_question", {})
    )
    scores = [
        score
        for item in by_question.values()
        if isinstance(item, dict)
        for score in [as_float(item.get("score"))]
        if score is not None
    ]
    return mean(scores) if scores else None


def summary_score(evaluation: dict[str, Any], metric: str) -> float | None:
    scorers = {
        "ctrs": ctrs_score,
        "wai": wai_score,
        "miti": miti_score,
        "alliance": alliance_score,
    }
    return scorers[metric](evaluation)


def summary_averages(
    evaluations_dir: Path, modes: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    averages_by_mode: dict[str, dict[str, float]] = {}
    counts_by_mode: dict[str, int] = {}

    for mode in modes:
        files = sorted((evaluations_dir / mode).glob("*.json"))
        counts_by_mode[mode] = len(files)
        scores: dict[str, list[float]] = {metric: [] for metric, _ in METRIC_COLUMNS}
        for path in files:
            evaluation = load_json(path)
            for metric, _ in METRIC_COLUMNS:
                score = summary_score(evaluation, metric)
                if score is not None:
                    scores[metric].append(score)

        averages_by_mode[mode] = {
            metric: mean(values)
            for metric, values in scores.items()
            if values
        }

    return averages_by_mode, counts_by_mode


def formatted_scores(averages: dict[str, float]) -> list[str]:
    return [
        f"{averages[skill]:.2f}" if skill in averages else "--"
        for skill, _ in SKILL_COLUMNS
    ]


def print_latex_table(mode: str, approach: str, averages: dict[str, float], count: int) -> None:
    values = formatted_scores(averages)
    title = MODE_LABELS.get(mode, mode.capitalize())
    print(f"% {title} clients, n={count}")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Therapist skill scores for {title} clients}}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("& \\multicolumn{3}{c}{CBT-specific Skills} & \\multicolumn{3}{c}{General Counseling Skills} \\\\")
    print("\\cline{2-4} \\cline{5-7}")
    print("Approach & Discovery & Focus & Strategy & Understanding & Interpersonal & Collaboration \\\\")
    print("\\hline")
    print(f"{approach} & {' & '.join(values)} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def print_markdown_table(mode: str, approach: str, averages: dict[str, float], count: int) -> None:
    values = formatted_scores(averages)
    print(f"### {MODE_LABELS.get(mode, mode.capitalize())} clients (n={count})")
    print("| Approach | Discovery | Focus | Strategy | Understanding | Interpersonal | Collaboration |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(f"| {approach} | {' | '.join(values)} |")
    print()


def summary_values(averages_by_mode: dict[str, dict[str, float]], modes: list[str]) -> list[str]:
    values: list[str] = []
    for mode in modes:
        averages = averages_by_mode.get(mode, {})
        values.extend(
            f"{averages[metric]:.2f}" if metric in averages else "--"
            for metric, _ in METRIC_COLUMNS
        )
    return values


def print_summary_latex_table(
    modes: list[str],
    approach: str,
    averages_by_mode: dict[str, dict[str, float]],
    counts_by_mode: dict[str, int],
) -> None:
    values = summary_values(averages_by_mode, modes)
    column_spec = "l" + "c" * (len(modes) * len(METRIC_COLUMNS))
    print("% Summary metrics by client difficulty")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Aggregate evaluation scores by client difficulty}")
    print(f"\\begin{{tabular}}{{{column_spec}}}")
    print("\\hline")
    group_headers = [
        f"\\multicolumn{{4}}{{c}}{{{MODE_LABELS.get(mode, mode.capitalize())}}}"
        for mode in modes
    ]
    print(f"Approach & {' & '.join(group_headers)} \\\\")
    clines = " ".join(
        f"\\cline{{{2 + index * 4}-{5 + index * 4}}}"
        for index, _ in enumerate(modes)
    )
    print(clines)
    metric_headers = [label for _mode in modes for _metric, label in METRIC_COLUMNS]
    print(f"& {' & '.join(metric_headers)} \\\\")
    print("\\hline")
    print(f"{approach} & {' & '.join(values)} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    counts = ", ".join(
        f"{MODE_LABELS.get(mode, mode.capitalize())} n={counts_by_mode.get(mode, 0)}"
        for mode in modes
    )
    print(f"% {counts}")
    print("\\end{table}")
    print()


def print_summary_markdown_table(
    modes: list[str],
    approach: str,
    averages_by_mode: dict[str, dict[str, float]],
    counts_by_mode: dict[str, int],
) -> None:
    values = summary_values(averages_by_mode, modes)
    headers = [
        f"{MODE_LABELS.get(mode, mode.capitalize())} {label}"
        for mode in modes
        for _metric, label in METRIC_COLUMNS
    ]
    counts = ", ".join(
        f"{MODE_LABELS.get(mode, mode.capitalize())} n={counts_by_mode.get(mode, 0)}"
        for mode in modes
    )
    print(f"### Summary metrics ({counts})")
    print(f"| Approach | {' | '.join(headers)} |")
    print(f"|---|{'|'.join(['---:'] * len(headers))}|")
    print(f"| {approach} | {' | '.join(values)} |")
    print()


def main() -> None:
    args = parse_args()
    if "therapist-skills" in args.tables:
        for mode in args.modes:
            averages, count = mode_averages(args.evaluations_dir, mode)
            if args.format == "latex":
                print_latex_table(mode, args.approach, averages, count)
            else:
                print_markdown_table(mode, args.approach, averages, count)

    if "summary" in args.tables:
        averages_by_mode, counts_by_mode = summary_averages(args.evaluations_dir, args.modes)
        if args.format == "latex":
            print_summary_latex_table(
                args.modes, args.approach, averages_by_mode, counts_by_mode
            )
        else:
            print_summary_markdown_table(
                args.modes, args.approach, averages_by_mode, counts_by_mode
            )


if __name__ == "__main__":
    main()
