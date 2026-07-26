"""Hybrid evaluator: rule-based signals, embedding similarity, and a judge.

This module is Algorithm 2 lines 5-15 for a single trial. It computes the
visible-acceptance signal and the auxiliary signals, calls the judge, and hands
the result to score_record, which owns the staged label rules. Service-level
blocks are judged too: the deterministic block rule runs after the judge call
and overrides its output, so the judge sees one call per target response
regardless of how the block signal came out.

The similarity signals are advisory throughout; scorers.py states the labeling
policy they fall under and implements the staged rules themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import JudgeCall, JudgeDecision, apply_judge_decision, run_judge
from .lexical import clean_text, detect_service_block
from .schema import Record
from .scorers import ScoreConfig, coerce_bool, score_record
from .semantic import SimilarityBackend, SimilaritySignals, get_backend


def similarity_signals(backend: SimilarityBackend, record: Record) -> SimilaritySignals:
    """Compute the advisory similarity signals for one record.

    Shared by the integrated scoring path and the offline judge-request
    builder, so both hand the judge the same numbers.
    """
    original = clean_text(record.get("original_prompt"))
    response = clean_text(record.get("model_response"))
    return backend.signals(original, response)


@dataclass
class HybridEvaluator:
    """Score records with the paper's hybrid labeling procedure.

    similarity defaults to the "auto" backend: embeddings when installed,
    difflib otherwise. A record is labeled by the first of these that applies:
    the stage labels it already carries, a decision in ``decisions`` keyed by
    record_id, or a call to ``judge_call``. With none of them, only records the
    block rule settles can be scored.

    judge_call maps a judge prompt to the judge's raw text reply. ``decisions``
    carries replies collected separately, which is the offline path behind
    ``rogueprompt-evaluate score --judge-decisions``.
    """

    similarity: SimilarityBackend | None = None
    judge_call: JudgeCall | None = None
    decisions: dict[str, JudgeDecision] | None = None
    config: ScoreConfig = field(default_factory=ScoreConfig)

    def __post_init__(self) -> None:
        if self.similarity is None:
            self.similarity = get_backend("auto")

    def _signals(self, record: Record) -> SimilaritySignals:
        return similarity_signals(self.similarity, record)

    def _decision(self, record: Record) -> JudgeDecision | None:
        if not self.decisions:
            return None
        return self.decisions.get(str(record.get("record_id")))

    def score(self, record: Record) -> Record:
        """Return a copy of the record with the staged score fields attached."""
        # Computed once and threaded through: detect_service_block warns on a
        # retried status, and recomputing it would repeat the warning.
        block = detect_service_block(record)
        signals = self._signals(record)

        working = dict(record)
        # Same keys and rounding as the judge prompt carries, so a signal reads
        # the same on the scored record and in the request that labeled it.
        working.update(signals.as_dict())

        labeled = (
            coerce_bool(record.get("reconstruction_success")) is not None
            and coerce_bool(record.get("execution_success")) is not None
        )
        if not labeled:
            decision = self._decision(record)
            if decision is None and self.judge_call is not None:
                decision = run_judge(
                    record, self.judge_call, signals=signals, block=block
                )
            if decision is not None:
                working = apply_judge_decision(
                    working, decision, strict=self.config.strict_judge, block=block
                )

        return score_record(working, config=self.config, block=block)

    def score_records(self, records: list[Record]) -> list[Record]:
        return [self.score(record) for record in records]
