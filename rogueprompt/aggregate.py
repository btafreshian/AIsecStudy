"""Aggregate scored RoguePrompt records."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Record
from .scorers import FAILURE_MODES


DEFAULT_GROUP_BY = ("method", "model")
STAGES = ("bypass", "reconstruction", "execution")


def _percent(numerator: int, denominator: int, digits: int = 2) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, digits)


def aggregate_scores(
    records: Iterable[Record],
    group_by: tuple[str, ...] = DEFAULT_GROUP_BY,
    digits: int = 2,
) -> list[Record]:
    """Aggregate staged success rates and failure-mode counts."""
    groups: dict[tuple[object, ...], list[Record]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) for field in group_by)
        groups[key].append(record)

    rows: list[Record] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        total = len(items)
        row: Record = {field: value for field, value in zip(group_by, key)}
        row["n"] = total

        for stage in STAGES:
            field = f"{stage}_success"
            successes = sum(record.get(field) is True for record in items)
            missing = sum(record.get(field) is None for record in items)
            row[f"{stage}_success_count"] = successes
            row[f"{stage}_missing_count"] = missing
            row[f"{stage}_pct"] = _percent(successes, total, digits)

        for mode in FAILURE_MODES:
            count = sum(record.get("failure_mode") == mode for record in items)
            row[f"{mode.lower()}_count"] = count
            row[f"{mode.lower()}_pct"] = _percent(count, total, digits)

        rows.append(row)

    return rows


def write_summary_csv(rows: list[Record], path: str | Path) -> None:
    """Write aggregate rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
