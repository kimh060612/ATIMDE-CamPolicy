#!/usr/bin/env python3
"""Aggregate clean MDE metrics from one or more experiment CSV files.

Methodology
-----------
1. Keep ``record_type == capture`` when that column exists.
2. Keep ``selected == 1`` when that column exists.  Logs without a
   ``selected`` column use every evaluated capture.
3. Drop rows whose AbsRel or A1 is missing/non-finite.
4. Mark a row as an outlier only when both conditions hold:
      AbsRel >= 0.5 AND A1 >= 0.60
5. Report per-run frame statistics, run-level mean/sample SD, and pooled
   frame statistics.  Lower AbsRel and higher A1 are better.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ABS_REL_OUTLIER_MIN = 0.4
A1_OUTLIER_MIN = 0.60
TRUE_VALUES = {"1", "1.0", "true", "yes"}


@dataclass(frozen=True)
class MetricRow:
    source: Path
    raw: Mapping[str, str]
    abs_rel: float
    a1: float


@dataclass(frozen=True)
class Stats:
    mean: float
    sd: float | None


@dataclass(frozen=True)
class RunResult:
    source: Path
    capture_count: int
    candidate_count: int
    valid_count: int
    clean: tuple[MetricRow, ...]
    outliers: tuple[MetricRow, ...]
    raw_abs_rel: Stats
    raw_a1: Stats
    clean_abs_rel: Stats
    clean_a1: Stats
    selection_mode: str


def sample_stats(values: Iterable[float]) -> Stats:
    data = list(values)
    if not data:
        raise ValueError("cannot calculate statistics for an empty sample")
    sd = statistics.stdev(data) if len(data) >= 2 else None
    return Stats(statistics.mean(data), sd)


def is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def read_run(path: Path) -> RunResult:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"abs_rel", "a1"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        columns = set(reader.fieldnames)

    captures = [
        row for row in rows
        if "record_type" not in columns or row.get("record_type") == "capture"
    ]
    has_selected = "selected" in columns
    candidates = (
        [row for row in captures if is_true(row.get("selected"))]
        if has_selected else captures
    )

    valid: list[MetricRow] = []
    for row in candidates:
        try:
            abs_rel = float(row["abs_rel"])
            a1 = float(row["a1"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(abs_rel) and math.isfinite(a1):
            valid.append(MetricRow(path, row, abs_rel, a1))

    if not valid:
        raise ValueError(f"{path}: no valid evaluated rows after filtering")

    outliers = tuple(
        row for row in valid
        if row.abs_rel >= ABS_REL_OUTLIER_MIN and row.a1 >= A1_OUTLIER_MIN
    )
    clean = tuple(row for row in valid if row not in outliers)
    if not clean:
        raise ValueError(f"{path}: every valid row was classified as an outlier")

    return RunResult(
        source=path,
        capture_count=len(captures),
        candidate_count=len(candidates),
        valid_count=len(valid),
        clean=clean,
        outliers=outliers,
        raw_abs_rel=sample_stats(row.abs_rel for row in valid),
        raw_a1=sample_stats(row.a1 for row in valid),
        clean_abs_rel=sample_stats(row.abs_rel for row in clean),
        clean_a1=sample_stats(row.a1 for row in clean),
        selection_mode="selected=1" if has_selected else "all captures",
    )


def fmt(stats: Stats) -> str:
    if stats.sd is None:
        return f"{stats.mean:.5f}"
    return f"{stats.mean:.5f} ± {stats.sd:.5f}"


def field(row: MetricRow, name: str) -> str:
    return row.raw.get(name, "") or "-"


def print_report(results: list[RunResult]) -> None:
    print(
        f"Outlier rule: AbsRel >= {ABS_REL_OUTLIER_MIN} "
        f"AND A1 >= {A1_OUTLIER_MIN:.2f}\n"
    )
    print("| Run | Filter | Candidates | Invalid | Outliers | Clean N | AbsRel | A1 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        invalid = result.candidate_count - result.valid_count
        print(
            f"| {result.source.name} | {result.selection_mode} "
            f"| {result.candidate_count} | {invalid} | {len(result.outliers)} "
            f"| {len(result.clean)} | {fmt(result.clean_abs_rel)} "
            f"| {fmt(result.clean_a1)} |"
        )

    pooled = [row for result in results for row in result.clean]
    pooled_abs_rel = sample_stats(row.abs_rel for row in pooled)
    pooled_a1 = sample_stats(row.a1 for row in pooled)
    print("\nAggregate")
    print("| Unit | N | AbsRel | A1 |")
    print("|---|---:|---:|---:|")
    if len(results) >= 2:
        run_abs_rel = sample_stats(result.clean_abs_rel.mean for result in results)
        run_a1 = sample_stats(result.clean_a1.mean for result in results)
        print(f"| Run means | {len(results)} | {fmt(run_abs_rel)} | {fmt(run_a1)} |")
    print(f"| Pooled frames | {len(pooled)} | {fmt(pooled_abs_rel)} | {fmt(pooled_a1)} |")

    outliers = [row for result in results for row in result.outliers]
    print(f"\nExcluded outliers: {len(outliers)}")
    if outliers:
        print("| Run | Round | Capture | Motion | Cell | AbsRel | A1 |")
        print("|---|---:|---:|---|---|---:|---:|")
        for row in outliers:
            print(
                f"| {row.source.name} | {field(row, 'round_index')} "
                f"| {field(row, 'capture_index')} | {field(row, 'motion_label')} "
                f"| {field(row, 'cell_id')} | {row.abs_rel:.5f} | {row.a1:.5f} |"
            )

    raw_abs_rel = statistics.mean(result.raw_abs_rel.mean for result in results)
    clean_abs_rel = statistics.mean(result.clean_abs_rel.mean for result in results)
    raw_a1 = statistics.mean(result.raw_a1.mean for result in results)
    clean_a1 = statistics.mean(result.clean_a1.mean for result in results)
    abs_improvement = 100.0 * (raw_abs_rel - clean_abs_rel) / raw_abs_rel
    a1_improvement = 100.0 * (clean_a1 - raw_a1) / raw_a1
    print(
        "\nOutlier-removal effect on the mean of run means: "
        f"AbsRel {raw_abs_rel:.5f} -> {clean_abs_rel:.5f} "
        f"({abs_improvement:+.2f}% improvement), "
        f"A1 {raw_a1:.5f} -> {clean_a1:.5f} "
        f"({a1_improvement:+.2f}% improvement)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate selected MDE results after combined AbsRel/A1 outlier removal."
    )
    parser.add_argument("csv", nargs="+", type=Path, help="one or more experiment CSV files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = [read_run(path) for path in args.csv]
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

