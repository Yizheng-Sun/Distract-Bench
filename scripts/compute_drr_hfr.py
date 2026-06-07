#!/usr/bin/env python3
"""Compute DRR/HFR from distractor-reference and correctness judgments.

Definitions over samples that have both judgments:
  DRR = P(reference)
  HFR = P(reference and incorrect)

The script supports either:
  1. separate reference and correctness judgment files/directories, or
  2. a combined file where each row contains both judgments.

Input files may be JSONL or CSV. Directory inputs are searched recursively.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GENERIC_DIR_NAMES = {
    "corrupt_results",
    "corruption_reference",
    "reference_judgments",
    "correctness_judgments",
    "judged_results",
    "examples",
    "outputs",
    "data",
}

DEFAULT_JOIN_FIELDS = ("final_id", "doc_id", "id", "question_id", "sample_id")
REFERENCE_FIELDS = (
    "reference_verdict",
    "distractor_reference",
    "references_distractor",
    "ref_verdict",
    "drr_verdict",
)
CORRECTNESS_FIELDS = (
    "correctness_verdict",
    "answer_verdict",
    "is_correct",
    "correct",
    "accuracy",
)


@dataclass
class Stats:
    model: str
    n: int = 0
    ref_correct: int = 0
    ref_wrong: int = 0
    no_ref_correct: int = 0
    no_ref_wrong: int = 0
    missing_correctness: int = 0
    missing_reference: int = 0

    @property
    def ref_total(self) -> int:
        return self.ref_correct + self.ref_wrong

    @property
    def no_ref_total(self) -> int:
        return self.no_ref_correct + self.no_ref_wrong

    @property
    def correct_total(self) -> int:
        return self.ref_correct + self.no_ref_correct

    @property
    def wrong_total(self) -> int:
        return self.ref_wrong + self.no_ref_wrong

    @property
    def drr(self) -> float:
        return safe_div(self.ref_total, self.n)

    @property
    def hfr(self) -> float:
        return safe_div(self.ref_wrong, self.n)

    @property
    def rrr(self) -> float:
        return safe_div(self.ref_correct, self.ref_total)

    @property
    def accuracy(self) -> float:
        return safe_div(self.correct_total, self.n)

    @property
    def failure_given_reference(self) -> float:
        return safe_div(self.ref_wrong, self.ref_total)

    @property
    def reference_given_failure(self) -> float:
        return safe_div(self.ref_wrong, self.wrong_total)

    @property
    def reference_given_correct(self) -> float:
        return safe_div(self.ref_correct, self.correct_total)


def safe_div(num: int, den: int) -> float:
    return float("nan") if den == 0 else num / den


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Distract-Bench DRR/HFR metrics from judged outputs.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        nargs="*",
        default=[],
        help="Reference-judgment JSONL/CSV files or directories. Verdicts are yes/no.",
    )
    parser.add_argument(
        "--correctness",
        type=Path,
        nargs="*",
        default=[],
        help="Correctness-judgment JSONL/CSV files or directories. Verdicts are correct/incorrect.",
    )
    parser.add_argument(
        "--combined",
        type=Path,
        nargs="*",
        default=[],
        help="Combined JSONL/CSV files or directories containing both judgments per row.",
    )
    parser.add_argument(
        "--join-key",
        default="auto",
        help=(
            "Field used to join separate judgments. Use 'auto' to try "
            f"{', '.join(DEFAULT_JOIN_FIELDS)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory for per-model metrics and summary outputs.",
    )
    return parser.parse_args()


def discover_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"missing path: {path}")
    return sorted(files)


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)
        return

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            yield row


def infer_model(row: dict[str, Any], path: Path) -> str:
    model = row.get("model")
    if model not in (None, ""):
        return str(model)
    parent = path.parent.name
    if parent and parent not in GENERIC_DIR_NAMES:
        return parent
    return "all"


def normalize_bool(value: Any, positive: set[str], negative: set[str]) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in positive:
        return True
    if text in negative:
        return False
    return None


def reference_value(row: dict[str, Any]) -> bool | None:
    for field in REFERENCE_FIELDS:
        if field in row:
            value = normalize_bool(row[field], {"yes", "y", "true", "1"}, {"no", "n", "false", "0"})
            if value is not None:
                return value

    verdict = normalize_bool(row.get("verdict"), {"yes", "y"}, {"no", "n"})
    return verdict


def correctness_value(row: dict[str, Any]) -> bool | None:
    for field in CORRECTNESS_FIELDS:
        if field in row:
            value = normalize_bool(
                row[field],
                {"correct", "yes", "y", "true", "1"},
                {"incorrect", "wrong", "no", "n", "false", "0"},
            )
            if value is not None:
                return value

    verdict = normalize_bool(row.get("verdict"), {"correct"}, {"incorrect", "wrong"})
    return verdict


def join_keys(row: dict[str, Any], join_key: str) -> list[tuple[str, str]]:
    fields = DEFAULT_JOIN_FIELDS if join_key == "auto" else (join_key,)
    keys: list[tuple[str, str]] = []
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            keys.append((field, str(value)))
    return keys


def add_observation(stats: Stats, referenced: bool | None, correct: bool | None) -> None:
    if referenced is None:
        stats.missing_reference += 1
        return
    if correct is None:
        stats.missing_correctness += 1
        return

    stats.n += 1
    if referenced and correct:
        stats.ref_correct += 1
    elif referenced and not correct:
        stats.ref_wrong += 1
    elif not referenced and correct:
        stats.no_ref_correct += 1
    else:
        stats.no_ref_wrong += 1


def load_correctness_index(
    paths: list[Path],
    join_key: str,
) -> dict[tuple[str, tuple[str, str]], bool]:
    index: dict[tuple[str, tuple[str, str]], bool] = {}
    for path in paths:
        for row in read_rows(path):
            correct = correctness_value(row)
            if correct is None:
                continue
            model = infer_model(row, path)
            for key in join_keys(row, join_key):
                index[(model, key)] = correct
                index.setdefault(("all", key), correct)
    return index


def lookup_correctness(
    index: dict[tuple[str, tuple[str, str]], bool],
    model: str,
    keys: list[tuple[str, str]],
) -> bool | None:
    for key in keys:
        if (model, key) in index:
            return index[(model, key)]
    for key in keys:
        if ("all", key) in index:
            return index[("all", key)]
    return None


def metrics_row(stats: Stats) -> dict[str, Any]:
    return {
        "model": stats.model,
        "n": stats.n,
        "ref_total": stats.ref_total,
        "no_ref_total": stats.no_ref_total,
        "ref_correct": stats.ref_correct,
        "ref_wrong": stats.ref_wrong,
        "no_ref_correct": stats.no_ref_correct,
        "no_ref_wrong": stats.no_ref_wrong,
        "accuracy": stats.accuracy,
        "drr": stats.drr,
        "hfr": stats.hfr,
        "rrr": stats.rrr,
        "failure_given_reference": stats.failure_given_reference,
        "reference_given_failure": stats.reference_given_failure,
        "reference_given_correct": stats.reference_given_correct,
        "missing_reference": stats.missing_reference,
        "missing_correctness": stats.missing_correctness,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if value != value else f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "n",
        "ref_total",
        "no_ref_total",
        "ref_correct",
        "ref_wrong",
        "no_ref_correct",
        "no_ref_wrong",
        "accuracy",
        "drr",
        "hfr",
        "rrr",
        "failure_given_reference",
        "reference_given_failure",
        "reference_given_correct",
        "missing_reference",
        "missing_correctness",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def add_stats(
    per_model: dict[str, Stats],
    model: str,
    referenced: bool | None,
    correct: bool | None,
) -> None:
    model_stats = per_model.setdefault(model, Stats(model=model))
    add_observation(model_stats, referenced, correct)


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'model':<42} {'n':>6} {'acc':>8} {'DRR':>8} {'HFR':>8} {'RRR':>8}")
    for row in rows:
        pct = lambda value: "-" if value != value else f"{100 * value:.1f}%"
        print(
            f"{row['model']:<42} {row['n']:>6} "
            f"{pct(row['accuracy']):>8} {pct(row['drr']):>8} "
            f"{pct(row['hfr']):>8} {pct(row['rrr']):>8}"
        )


def main() -> None:
    args = parse_args()
    if not args.combined and not (args.reference and args.correctness):
        raise SystemExit("provide either --combined or both --reference and --correctness")

    per_model: dict[str, Stats] = {}

    for path in discover_files(args.combined):
        for row in read_rows(path):
            referenced = reference_value(row)
            correct = correctness_value(row)
            if referenced is None and correct is None:
                continue
            add_stats(
                per_model,
                infer_model(row, path),
                referenced,
                correct,
            )

    if args.reference or args.correctness:
        reference_files = discover_files(args.reference)
        correctness_files = discover_files(args.correctness)
        correctness_index = load_correctness_index(correctness_files, args.join_key)
        for path in reference_files:
            for row in read_rows(path):
                model = infer_model(row, path)
                keys = join_keys(row, args.join_key)
                referenced = reference_value(row)
                if referenced is None and not keys:
                    continue
                correct = lookup_correctness(correctness_index, model, keys) if keys else None
                add_stats(
                    per_model,
                    model,
                    referenced,
                    correct,
                )

    model_rows = sorted((metrics_row(row) for row in per_model.values()), key=lambda row: str(row["model"]))
    if not model_rows or not any(row["n"] for row in model_rows):
        raise SystemExit("no rows with both reference and correctness judgments were joined")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_model_metrics.csv", model_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            json_ready({
                "definitions": {
                    "drr": "P(distractor reference)",
                    "hfr": "P(distractor reference and incorrect answer)",
                    "rrr": "P(correct answer | distractor reference)",
                },
                "per_model": model_rows,
            }),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print_table(model_rows)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
