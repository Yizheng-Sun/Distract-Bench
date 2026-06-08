#!/usr/bin/env python3
"""LLM judge for Distract-Bench distractor-reference labels.

This script regenerates the reference judgments used for DRR/HFR. For each
corrupt-side model output, it asks an OpenAI model whether the response
explicitly references, mentions, quotes, attends to, or uses the injected
semantic distractor.

The judge does not grade answer correctness. It only produces the
distractor-reference verdict:

  yes: the model output explicitly uses or mentions the injected distractor
  no:  the model output ignores the injected distractor

Usage:
  export OPENAI_API_KEY=...
  python scripts/judge_distractor_reference.py --dry-run --limit 2
  python scripts/judge_distractor_reference.py --workers 4
  python scripts/judge_distractor_reference.py --only-model Qwen__Qwen2.5-VL-7B-Instruct

By default outputs are written under outputs/corruption_reference/. To overwrite
or regenerate the released reference files, pass:

  --out-root model_outputs/corruption_reference --rerun
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Any, **_kwargs: Any) -> Any:
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "model_outputs" / "corrupt_results"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "corruption_reference"

DEFAULT_JUDGE_MODEL = "gpt-5-nano"
DEFAULT_WORKERS = 4
DEFAULT_MAX_TOKENS = 1200
DEFAULT_MAX_OUTPUT_CHARS = 30000
DEFAULT_OUTPUT_FIELDS = ("reps", "resps", "filtered_reps", "filtered_resps")

SYSTEM_PROMPT = """You are evaluating Distraction Reference Rate (DRR) for Distract-Bench.

You will receive the original task, the semantic distractor injected into the
input image, the detailed image edit instruction used to render that
distractor, and the model's output including any reasoning chain.

Your job is NOT to grade answer correctness. Decide only whether the model
output explicitly references, mentions, quotes, attends to, or uses the injected
distractor in its reasoning or final answer.

Return YES when the output:
- names or quotes a label, number, object, note, arrow, box, highlight, or other
  content that was introduced by the distractor;
- uses a value or claim from the distractor as evidence in the solution;
- mentions the red annotation, added callout, highlighted region, or visual mark;
- follows the described trap reasoning path, when the text makes that reliance
  explicit.

Return NO when the output:
- solves the original task without mentioning or using the distractor;
- mentions only information that is part of the original question or normal
  solution;
- gives the distractor's trap answer but does not explicitly refer to the
  distractor or its introduced evidence;
- is ambiguous. Prefer NO unless there is clear textual evidence in the output.

Return strict JSON only:
{"verdict":"yes"|"no","reason":"one short sentence","evidence":"short quote or empty string","confidence":0.0}
"""


@dataclass(frozen=True)
class EditInfo:
    final_id: str
    image: str
    dead_end_description: str
    trap_reasoning_path: str
    trap_answer: str
    edit_instruction: str


@dataclass(frozen=True)
class QuestionInfo:
    final_id: str
    question: str
    options: list[str]
    answer: str


@dataclass(frozen=True)
class PendingRecord:
    row_index: int
    doc_id: Any
    final_id: str
    question: str
    gold: str
    output_field: str
    model_output: str
    model_output_sha1: str
    model_output_chars: int
    model_output_truncated: bool
    edit: EditInfo


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        flattened: list[str] = []

        def walk(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, list):
                for child in item:
                    walk(child)
                return
            flattened.append(str(item))

        walk(value)
        for item in flattened:
            if item.strip():
                return item
        return ""
    return str(value)


def get_nested(record: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = record
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def clean_stem(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            stem = clean_stem(item)
            if stem:
                return stem
        return None
    name = str(value).rsplit("/", 1)[-1]
    if not name:
        return None
    return name.rsplit(".", 1)[0] or None


def json_id_value(final_id: str) -> int | str:
    return int(final_id) if final_id.isdigit() else final_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_edits(data_dir: Path) -> dict[str, EditInfo]:
    path = data_dir / "edit_instructions.jsonl"
    if not path.exists():
        raise SystemExit(f"missing edit instructions: {path}")

    edits: dict[str, EditInfo] = {}
    for row in read_jsonl(path):
        final_id = str(row.get("final_id") or "").strip()
        if not final_id:
            continue
        edit = row.get("edit") or {}
        edits[final_id] = EditInfo(
            final_id=final_id,
            image=str(row.get("image") or f"images/{final_id}.png"),
            dead_end_description=str(edit.get("dead_end_description") or "").strip(),
            trap_reasoning_path=str(edit.get("trap_reasoning_path") or "").strip(),
            trap_answer=str(edit.get("trap_answer") or "").strip(),
            edit_instruction=str(edit.get("edit_instruction") or "").strip(),
        )
    return edits


def load_questions(data_dir: Path) -> dict[str, QuestionInfo]:
    path = data_dir / "questions.jsonl"
    if not path.exists():
        raise SystemExit(f"missing questions: {path}")

    questions: dict[str, QuestionInfo] = {}
    for row in read_jsonl(path):
        final_id = str(row.get("final_id") or "").strip()
        if not final_id:
            continue
        raw_options = row.get("options") or []
        if isinstance(raw_options, list):
            options = [str(item) for item in raw_options]
        else:
            options = [str(raw_options)]
        questions[final_id] = QuestionInfo(
            final_id=final_id,
            question=str(row.get("question") or "").strip(),
            options=options,
            answer=str(row.get("answer") or "").strip(),
        )
    return questions


def format_question(question: QuestionInfo | None, record: dict[str, Any]) -> str:
    if question is None:
        return coerce_text(
            get_nested(record, "llm_as_judge_eval.query")
            or get_nested(record, "submission.query")
            or record.get("input")
        )
    parts = [question.question]
    if question.options:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices = []
        for idx, option in enumerate(question.options):
            label = labels[idx] if idx < len(labels) else str(idx + 1)
            choices.append(f"({label}) {option}")
        parts.append("Choices:\n" + "\n".join(choices))
    return "\n\n".join(part for part in parts if part)


def select_model_output(record: dict[str, Any], output_fields: tuple[str, ...]) -> tuple[str, str]:
    for field in output_fields:
        text = coerce_text(get_nested(record, field))
        if text.strip():
            return field, text
    return "", ""


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def find_final_id(record: dict[str, Any], edits: dict[str, EditInfo]) -> str | None:
    candidates: list[str] = []
    for path in (
        "final_id",
        "doc_id",
        "llm_as_judge_eval.question_id",
        "submission.question_id",
        "question_id",
        "id",
    ):
        value = get_nested(record, path)
        if value is not None:
            candidates.append(str(value))

    stem = clean_stem(record.get("input_media"))
    if stem:
        candidates.append(stem)

    for candidate in candidates:
        if candidate in edits:
            return candidate
        if candidate.isdigit() and str(int(candidate)) in edits:
            return str(int(candidate))
    return None


def discover_files(source_root: Path, only_model: str | None) -> list[Path]:
    if not source_root.exists():
        raise SystemExit(f"missing source root: {source_root}")
    files: list[Path] = []
    for path in sorted(source_root.rglob("*samples*.jsonl")):
        rel = path.relative_to(source_root).parts
        if len(rel) < 2:
            continue
        model = rel[0]
        if only_model and model != only_model:
            continue
        files.append(path)
    return files


def model_from_path(source_root: Path, path: Path) -> str:
    return path.relative_to(source_root).parts[0]


def output_path(source_root: Path, out_root: Path, input_path: Path) -> Path:
    rel = input_path.relative_to(source_root)
    return out_root / rel.parent / f"{input_path.stem}_ref_judged.jsonl"


def load_seen_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for row in read_jsonl(path):
        verdict = row.get("verdict")
        if verdict not in {"yes", "no"}:
            continue
        if isinstance(row.get("row_index"), int):
            seen.add(f"row:{row['row_index']}")
        if row.get("final_id") not in (None, ""):
            seen.add(f"final:{row['final_id']}")
    return seen


def make_distraction_block(edit: EditInfo) -> str:
    parts = [
        ("Short distractor description", edit.dead_end_description),
        ("Detailed visual edit instruction", edit.edit_instruction),
        ("Trap reasoning path", edit.trap_reasoning_path),
        ("Trap answer", edit.trap_answer),
    ]
    return "\n".join(f"{label}: {text}" for label, text in parts if text)


def render_user_prompt(item: PendingRecord) -> str:
    trunc_note = ""
    if item.model_output_truncated:
        trunc_note = (
            f"\nNOTE: The model output was truncated to the first "
            f"{len(item.model_output)} of {item.model_output_chars} characters."
        )
    return (
        f"SAMPLE ID: {item.final_id}\n\n"
        f"ORIGINAL TASK:\n{item.question}\n\n"
        f"GOLD ANSWER:\n{item.gold}\n\n"
        f"INJECTED DISTRACTOR:\n{make_distraction_block(item.edit)}\n\n"
        f"MODEL OUTPUT FIELD: {item.output_field}{trunc_note}\n"
        f"MODEL OUTPUT:\n{item.model_output}\n\n"
        "Does the MODEL OUTPUT explicitly refer to or use the INJECTED "
        "DISTRACTOR? Return strict JSON only."
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict in {"true", "referenced", "reference"}:
        verdict = "yes"
    if verdict in {"false", "not_referenced", "none"}:
        verdict = "no"
    if verdict not in {"yes", "no"}:
        raise ValueError(f"unexpected verdict: {parsed!r}")

    confidence_raw = parsed.get("confidence", "")
    try:
        confidence: float | str = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = ""

    return {
        "verdict": verdict,
        "reason": str(parsed.get("reason", ""))[:700],
        "evidence": str(parsed.get("evidence", ""))[:500],
        "confidence": confidence,
    }


def create_response(
    client: Any,
    judge_model: str,
    max_tokens: int,
    timeout: float,
    reasoning_effort: str,
    prompt: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": judge_model,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_tokens,
        "timeout": timeout,
    }
    if reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    try:
        return client.responses.create(**kwargs)
    except Exception as exc:
        if reasoning_effort != "none" and "reasoning" in str(exc).lower():
            kwargs.pop("reasoning", None)
            return client.responses.create(**kwargs)
        raise


def call_judge(client: Any, args: argparse.Namespace, item: PendingRecord) -> dict[str, Any]:
    prompt = render_user_prompt(item)
    last_err: Exception | None = None
    for attempt in range(args.max_retries):
        try:
            resp = create_response(
                client=client,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                prompt=prompt,
            )
            text = (getattr(resp, "output_text", "") or "").strip()
            if not text:
                status = getattr(resp, "status", "") or "empty_output_text"
                details = getattr(resp, "incomplete_details", None)
                reason = getattr(details, "reason", "") if details is not None else ""
                raise ValueError(f"OpenAI returned no output text ({status}:{reason})")
            return parse_judge_json(text)
        except Exception as exc:
            last_err = exc
            if attempt + 1 < args.max_retries:
                time.sleep(min(args.retry_sleep_max, 1.5 * (2 ** attempt)))
    return {
        "verdict": "parse_error",
        "reason": f"{type(last_err).__name__}: {last_err}"[:700],
        "evidence": "",
        "confidence": "",
    }


def judge_one(client: Any, args: argparse.Namespace, model: str, item: PendingRecord) -> dict[str, Any]:
    result = call_judge(client, args, item)
    return {
        "model": model,
        "row_index": item.row_index,
        "doc_id": json_id_value(item.final_id),
        "final_id": item.final_id,
        "output_field": item.output_field,
        "model_output_sha1": item.model_output_sha1,
        "model_output_chars": item.model_output_chars,
        "model_output_truncated": item.model_output_truncated,
        "distraction": item.edit.dead_end_description,
        "edit_instruction": item.edit.edit_instruction,
        "trap_answer": item.edit.trap_answer,
        **result,
        "image": f"data/{item.edit.image}",
    }


def load_pending_records(
    input_path: Path,
    out_path: Path,
    edits: dict[str, EditInfo],
    questions: dict[str, QuestionInfo],
    args: argparse.Namespace,
) -> tuple[list[PendingRecord], dict[str, int]]:
    existing_seen = load_seen_keys(out_path)
    seen = set() if args.rerun else existing_seen
    stats = {
        "already_done": len(existing_seen),
        "skipped_no_edit": 0,
        "skipped_no_output": 0,
        "invalid_json": 0,
    }
    pending: list[PendingRecord] = []
    output_fields = tuple(field.strip() for field in args.output_fields.split(",") if field.strip())

    with input_path.open(encoding="utf-8") as fh:
        for row_index, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line, strict=False)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            final_id = find_final_id(record, edits)
            if f"row:{row_index}" in seen or (final_id and f"final:{final_id}" in seen):
                continue
            if not final_id or final_id not in edits:
                stats["skipped_no_edit"] += 1
                continue

            output_field, output = select_model_output(record, output_fields)
            if not output.strip():
                stats["skipped_no_output"] += 1
                continue

            output_sha1 = hashlib.sha1(output.encode("utf-8", errors="replace")).hexdigest()
            output_chars = len(output)
            output_for_judge, was_truncated = truncate_text(output, args.max_output_chars)
            question = questions.get(final_id)
            pending.append(
                PendingRecord(
                    row_index=row_index,
                    doc_id=record.get("doc_id"),
                    final_id=final_id,
                    question=format_question(question, record),
                    gold=question.answer if question is not None else coerce_text(record.get("target")),
                    output_field=output_field,
                    model_output=output_for_judge,
                    model_output_sha1=output_sha1,
                    model_output_chars=output_chars,
                    model_output_truncated=was_truncated,
                    edit=edits[final_id],
                )
            )
            if args.limit and len(pending) >= args.limit:
                break

    return pending, stats


def process_file(
    client: Any,
    source_root: Path,
    out_root: Path,
    input_path: Path,
    edits: dict[str, EditInfo],
    questions: dict[str, QuestionInfo],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = model_from_path(source_root, input_path)
    out_path = output_path(source_root, out_root, input_path)
    if args.rerun and out_path.exists() and not args.dry_run:
        out_path.unlink()

    pending, stats = load_pending_records(input_path, out_path, edits, questions, args)

    if args.dry_run:
        print(f"\n=== DRY RUN: {display_path(input_path)} ===")
        print(f"model: {model}")
        print(f"would judge: {len(pending)}")
        print(f"already done keys: {stats['already_done']}")
        print(f"skipped_no_edit: {stats['skipped_no_edit']}")
        print(f"skipped_no_output: {stats['skipped_no_output']}")
        print(f"invalid_json: {stats['invalid_json']}")
        for item in pending[:3]:
            print(f"--- row {item.row_index} doc {item.doc_id} final_id={item.final_id}")
            print(f"output_field={item.output_field} chars={item.model_output_chars} truncated={item.model_output_truncated}")
            print(f"distractor: {item.edit.dead_end_description[:180]}")
            print(f"output head: {item.model_output[:220]!r}")
        return {"model": model, "yes": 0, "no": 0, "errors": 0, **stats}

    yes = no = errors = 0
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_fh, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_one, client, args, model, item): item for item in pending}
        pbar = tqdm(as_completed(futures), total=len(futures), desc=model, leave=False)
        for future in pbar:
            row = future.result()
            with write_lock:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
            verdict = row.get("verdict")
            if verdict == "yes":
                yes += 1
            elif verdict == "no":
                no += 1
            else:
                errors += 1

    return {"model": model, "yes": yes, "no": no, "errors": errors, **stats}


def read_model_records(model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob("*_ref_judged.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def fmt_rate(value: float) -> str:
    return f"{value:.4f}" if value == value else ""


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%" if value == value else "-"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def aggregate_and_write(out_root: Path) -> list[dict[str, Any]]:
    if not out_root.exists():
        return []
    model_dirs = sorted(path for path in out_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        yes = no = errors = 0
        for row in read_model_records(model_dir):
            verdict = row.get("verdict")
            if verdict == "yes":
                yes += 1
            elif verdict == "no":
                no += 1
            else:
                errors += 1
        n_eval = yes + no
        drr = yes / n_eval if n_eval else float("nan")
        rows.append(
            {
                "model": model_dir.name,
                "n_total": yes + no + errors,
                "n_eval": n_eval,
                "yes": yes,
                "no": no,
                "errors": errors,
                "drr": drr,
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    per_model_csv = out_root / "per_model_rates.csv"
    with per_model_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "n_total", "n_eval", "yes", "no", "errors", "drr", "reference_rate"])
        for row in sorted(rows, key=lambda item: -(item["drr"] if item["drr"] == item["drr"] else -1)):
            writer.writerow(
                [
                    row["model"],
                    row["n_total"],
                    row["n_eval"],
                    row["yes"],
                    row["no"],
                    row["errors"],
                    fmt_rate(row["drr"]),
                    fmt_rate(row["drr"]),
                ]
            )

    report = out_root / "REPORT.md"
    total_yes = sum(row["yes"] for row in rows)
    total_no = sum(row["no"] for row in rows)
    total_errors = sum(row["errors"] for row in rows)
    total_eval = total_yes + total_no
    pooled_drr = total_yes / total_eval if total_eval else float("nan")

    md: list[str] = []
    md.append("# Distract-Bench DRR Judge Report\n")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    md.append(
        "DRR is the fraction of corrupt-side Distract-Bench outputs whose "
        "reasoning or answer explicitly refers to the injected semantic "
        "distractor. `drr = yes / (yes + no)`; parse errors are excluded from "
        "the denominator.\n"
    )
    md.append("## At a glance\n")
    md.append(f"- Models evaluated: {len(rows)}")
    md.append(f"- Total evaluated judgments: {total_eval}")
    md.append(f"- Yes: {total_yes}, No: {total_no}, Errors: {total_errors}")
    md.append(f"- Pooled DRR: {fmt_pct(pooled_drr)}\n")
    md.append("## Per-model DRR\n")
    table_rows = []
    for row in sorted(rows, key=lambda item: -(item["drr"] if item["drr"] == item["drr"] else -1)):
        table_rows.append(
            [
                row["model"],
                str(row["n_eval"]),
                str(row["yes"]),
                str(row["no"]),
                str(row["errors"]),
                fmt_pct(row["drr"]),
            ]
        )
    md.append(render_table(["Model", "n_eval", "yes", "no", "errors", "DRR"], table_rows))
    md.append("\n## Files\n")
    md.append("- `per_model_rates.csv`")
    md.append("- `<model>/*_ref_judged.jsonl`")
    md.append("")
    report.write_text("\n".join(md), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge whether Distract-Bench outputs reference distractors.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only-model", default=None, help="restrict to one model directory")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="cap records per input file")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=DEFAULT_MAX_OUTPUT_CHARS,
        help="truncate model output sent to judge; 0 means no truncation",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high"],
        help="OpenAI reasoning effort; use 'none' for non-reasoning judge models.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-max", type=float, default=30.0)
    parser.add_argument(
        "--output-fields",
        default=",".join(DEFAULT_OUTPUT_FIELDS),
        help="comma-separated field priority for model output",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="ignore existing per-sample judgments and regenerate output files",
    )
    parser.add_argument("--dry-run", action="store_true", help="show planned work, no API calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = resolve_path(args.source_root)
    data_dir = resolve_path(args.data_dir)
    out_root = resolve_path(args.out_root)

    load_dotenv(REPO_ROOT / ".env")

    edits = load_edits(data_dir)
    questions = load_questions(data_dir)
    files = discover_files(source_root, args.only_model)

    print(f"edit instructions: {len(edits)}")
    print(f"questions: {len(questions)}")
    print(f"input files: {len(files)}")
    print(f"judge model: {args.judge_model}")
    print(f"output fields: {args.output_fields}")
    print(f"output root: {display_path(out_root)}")
    for path in files:
        print(f"  - {display_path(path)}")

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. Put it in .env or export it.", file=sys.stderr)
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Missing dependency: pip install openai python-dotenv tqdm") from exc
        client = OpenAI()

    for path in files:
        summary = process_file(client, source_root, out_root, path, edits, questions, args)
        if args.dry_run:
            continue
        denom = summary["yes"] + summary["no"]
        rate = summary["yes"] / denom if denom else float("nan")
        tqdm.write(
            f"[{summary['model']}] yes={summary['yes']} no={summary['no']} "
            f"errors={summary['errors']} drr={fmt_pct(rate)}"
        )

    if args.dry_run:
        return

    rows = aggregate_and_write(out_root)
    print("\n=== Distract-Bench DRR per model ===")
    print(f"{'model':<40} {'n_eval':>6} {'yes':>5} {'no':>5} {'err':>5} {'DRR':>8}")
    for row in sorted(rows, key=lambda item: -(item["drr"] if item["drr"] == item["drr"] else -1)):
        print(
            f"{row['model']:<40} {row['n_eval']:>6} {row['yes']:>5} "
            f"{row['no']:>5} {row['errors']:>5} {fmt_pct(row['drr']):>8}"
        )
    print(f"\nWrote {display_path(out_root)}")


if __name__ == "__main__":
    main()
