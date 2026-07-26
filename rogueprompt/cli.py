"""CLI for RoguePrompt evaluation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .aggregate import aggregate_at3, aggregate_scores, write_summary_csv
from .evaluator import HybridEvaluator, similarity_signals
from .judge import build_judge_request, load_judge_decisions, openai_compatible_judge
from .schema import SchemaError, load_records, require_valid_records, validate_records
from .scorers import ScoreConfig
from .semantic import get_backend
from .versions import run_metadata, stamp_records


def _write_json(payload: object, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(records: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_by(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _add_similarity_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--similarity",
        choices=("auto", "jina", "difflib"),
        default="auto",
        help="similarity backend for the advisory judge signals (default: auto)",
    )


def validate_command(args: argparse.Namespace) -> int:
    records = load_records(args.input)
    errors = validate_records(records)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid: {len(records)} records")
    return 0


def score_command(args: argparse.Namespace) -> int:
    if args.judge_decisions and args.judge_endpoint:
        raise ValueError(
            "--judge-decisions and --judge-endpoint both supply the judge "
            "labels; pass one or the other"
        )

    records = require_valid_records(load_records(args.input))
    config = ScoreConfig(labels_only=args.labels_only, strict_judge=args.strict)

    judge_call = None
    if args.judge_endpoint:
        judge_call = openai_compatible_judge(
            args.judge_endpoint,
            args.judge_model,
            api_key_env=args.judge_api_key_env,
        )

    # One scoring path whether the judge is called here or its replies were
    # collected separately, so both label a given reply identically.
    evaluator = HybridEvaluator(
        similarity=get_backend(args.similarity),
        judge_call=judge_call,
        decisions=(
            load_judge_decisions(args.judge_decisions) if args.judge_decisions else None
        ),
        config=config,
    )
    scored = evaluator.score_records(records)

    if not args.no_version_stamp:
        # This run did the labeling, so it speaks for the evaluator version
        # only; generation versions are filled in but never overwritten.
        scored = stamp_records(scored)

    summary = aggregate_scores(scored, group_by=_group_by(args.group_by))

    if args.run_metadata:
        _write_json(run_metadata(), args.run_metadata)
    if args.output:
        _write_json(scored, args.output)
    if args.summary_json:
        _write_json(summary, args.summary_json)
    if args.summary_csv:
        write_summary_csv(summary, args.summary_csv)
    if args.conditions:
        write_summary_csv(aggregate_at3(scored), args.conditions)
    if not (args.output or args.summary_json or args.summary_csv or args.conditions):
        print(json.dumps({"summary": summary}, indent=2, ensure_ascii=False))

    return 0


def judge_requests_command(args: argparse.Namespace) -> int:
    records = require_valid_records(load_records(args.input))
    backend = get_backend(args.similarity)
    # One request per response, blocks included, carrying the same advisory
    # signals the integrated path computes.
    requests = [
        build_judge_request(record, signals=similarity_signals(backend, record))
        for record in records
    ]
    _write_jsonl(requests, args.output)
    print(f"wrote {len(requests)} judge requests to {args.output}")
    return 0


def versions_command(args: argparse.Namespace) -> int:
    metadata = run_metadata()
    if args.output:
        _write_json(metadata, args.output)
        print(f"wrote run metadata to {args.output}")
    else:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rogueprompt-evaluate",
        description="Validate, score, and summarize RoguePrompt evaluation records.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate an evaluation JSON/JSONL file")
    validate_parser.add_argument("input")
    validate_parser.set_defaults(func=validate_command)

    score_parser = subparsers.add_parser("score", help="score and aggregate evaluation records")
    score_parser.add_argument("input")
    score_parser.add_argument("--output", help="write scored records as JSON")
    score_parser.add_argument("--summary-json", help="write trial-level aggregate summary as JSON")
    score_parser.add_argument("--summary-csv", help="write trial-level aggregate summary as CSV")
    score_parser.add_argument("--conditions", help="write condition-level @3 CSV to this path")
    score_parser.add_argument("--group-by", default="method,model", help="comma-separated grouping fields")
    _add_similarity_argument(score_parser)
    score_parser.add_argument(
        "--judge-endpoint",
        help="base URL of an OpenAI-compatible judge (e.g. a self-hosted Llama-3.3-70B)",
    )
    score_parser.add_argument(
        "--judge-model",
        default="llama-3.3-70b-instruct",
        help="model name passed to the judge endpoint",
    )
    score_parser.add_argument(
        "--judge-api-key-env",
        default="ROGUEPROMPT_JUDGE_API_KEY",
        help="environment variable holding the judge API key",
    )
    score_parser.add_argument(
        "--judge-decisions",
        help="JSON/JSONL decisions from an external LLM judge; not combinable "
        "with --judge-endpoint",
    )
    score_parser.add_argument(
        "--labels-only",
        action="store_true",
        help="leave an accepted response unlabeled when no judge decision reached it, instead of stopping",
    )
    score_parser.add_argument(
        "--strict",
        action="store_true",
        help="stop on a judge BI for an accepted response instead of relabeling it OTH",
    )
    score_parser.add_argument("--run-metadata", help="write this run's component versions and digests as JSON")
    score_parser.add_argument(
        "--no-version-stamp",
        action="store_true",
        help="do not add the 'versions' and 'configuration_id' fields to scored records",
    )
    score_parser.set_defaults(func=score_command)

    judge_parser = subparsers.add_parser("judge-requests", help="write judge requests as JSONL")
    judge_parser.add_argument("input")
    judge_parser.add_argument("--output", required=True)
    _add_similarity_argument(judge_parser)
    judge_parser.set_defaults(func=judge_requests_command)

    versions_parser = subparsers.add_parser(
        "versions", help="print the per-component versions recorded in logs"
    )
    versions_parser.add_argument("--output", help="write the run metadata as JSON")
    versions_parser.set_defaults(func=versions_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SchemaError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
