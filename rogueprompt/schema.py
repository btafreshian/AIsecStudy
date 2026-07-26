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


class SchemaError(ValueError):
    """Raised when an evaluation file does not match the expected record shape."""


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file, tolerating a leading byte-order mark.

    Every input file here comes from the user, and Windows tooling routinely
    writes UTF-8 with a BOM. utf-8-sig drops that BOM when it is there and is
    identical to utf-8 when it is not, so those files load without a manual
    re-encode. Outputs are still written as plain UTF-8.
    """
    return Path(path).read_text(encoding="utf-8-sig")


def load_records(path: str | Path) -> list[Record]:
    """Load records from a JSON list, a {"records": [...]} object, or JSONL."""
    input_path = Path(path)
    text = read_text(input_path)

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

    errors.extend(_validate_status_signals(record, prefix))
    errors.extend(_validate_provenance(record, prefix))

    return errors


# The "observable status/error signals" of Section 4.5, which the block check
# in lexical.py reads. All optional: a record may carry none of them.
STATUS_TEXT_FIELDS = (
    "finish_reason",
    "stop_reason",
    "block_reason",
    "error_code",
    "error_message",
    "error",
)


def _validate_status_signals(record: Record, prefix: str) -> list[str]:
    """Check the optional provider status and error fields."""
    errors: list[str] = []

    status = record.get("status_code")
    if status is not None and (not isinstance(status, int) or isinstance(status, bool)):
        errors.append(f"{prefix}field 'status_code' must be an integer when present")

    for field in STATUS_TEXT_FIELDS:
        if field in record and record[field] is not None and not isinstance(record[field], str):
            errors.append(f"{prefix}field {field!r} must be a string when present")

    return errors


def _validate_provenance(record: Record, prefix: str) -> list[str]:
    """Check the optional Section 4.5 version block.

    Optional because records are collected by the user, and a run predating
    the version block is still a valid record. The check is structural only:
    which component names are meaningful is versions.py's business, and this
    module stays below it so that module can import this one.
    """
    errors: list[str] = []

    if "configuration_id" in record and not isinstance(record["configuration_id"], str):
        errors.append(f"{prefix}field 'configuration_id' must be a string when present")

    versions = record.get("versions")
    if versions is None:
        return errors
    if not isinstance(versions, dict):
        errors.append(f"{prefix}field 'versions' must be an object when present")
    elif not all(
        isinstance(name, str) and isinstance(value, str) for name, value in versions.items()
    ):
        errors.append(f"{prefix}field 'versions' must map component names to version strings")

    return errors


def validate_records(records: list[Record]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(validate_record(record, index=index))
    return errors


def require_valid_records(records: list[Record]) -> list[Record]:
    """Validate records, raising SchemaError if any of them fail."""
    errors = validate_records(records)
    if errors:
        raise SchemaError("\n".join(errors))
    return records
