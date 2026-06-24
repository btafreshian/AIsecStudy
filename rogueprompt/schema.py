"""Schema helpers for RoguePrompt evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Record = dict[str, Any]

REQUIRED_FIELDS = (
    "record_id",
    "prompt_index",
    "category",
    "method",
    "model",
    "original_prompt",
    "transformed_prompt",
)

OPTIONAL_RAW_FIELDS = (
    "model_response",
    "blocked",
    "refused",
    "reconstructed_text",
    "judge_notes",
)

OUTPUT_FIELDS = (
    "bypass_success",
    "reconstruction_success",
    "execution_success",
    "failure_mode",
)


class SchemaError(ValueError):
    """Raised when an evaluation file does not match the expected record shape."""


def load_records(path: str | Path) -> list[Record]:
    """Load records from a JSON list, ``{"records": [...]}``, or JSONL file."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")

    if input_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            records = payload

    if not isinstance(records, list):
        raise SchemaError(f"Expected a list of records in {input_path}")
    if not all(isinstance(record, dict) for record in records):
        raise SchemaError(f"Every item in {input_path} must be a JSON object")

    return records


def validate_record(record: Record, index: int | None = None) -> list[str]:
    """Return schema errors for one evaluation record."""
    prefix = f"record {index}: " if index is not None else ""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"{prefix}missing required field {field!r}")
        elif record[field] in (None, ""):
            errors.append(f"{prefix}field {field!r} cannot be empty")

    if "prompt_index" in record and not isinstance(record["prompt_index"], int):
        errors.append(f"{prefix}field 'prompt_index' must be an integer")

    for field in ("record_id", "category", "method", "model"):
        if field in record and not isinstance(record[field], str):
            errors.append(f"{prefix}field {field!r} must be a string")

    for field in ("original_prompt", "transformed_prompt", "model_response", "reconstructed_text"):
        if field in record and record[field] is not None and not isinstance(record[field], str):
            errors.append(f"{prefix}field {field!r} must be a string when present")

    return errors


def validate_records(records: list[Record]) -> list[str]:
    """Return all schema errors for a list of records."""
    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(validate_record(record, index=index))
    return errors


def require_valid_records(records: list[Record]) -> list[Record]:
    """Validate records and raise ``SchemaError`` on the first schema failure."""
    errors = validate_records(records)
    if errors:
        raise SchemaError("\n".join(errors))
    return records
