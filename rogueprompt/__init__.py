"""RoguePrompt utilities."""

from .aggregate import aggregate_scores
from .judge import JudgeDecision, build_judge_request, build_judge_prompt, parse_judge_decision
from .schema import REQUIRED_FIELDS, SchemaError, load_records, require_valid_records, validate_record
from .scorers import (
    FAILURE_MODES,
    ScoreConfig,
    determine_failure_mode,
    score_bypass,
    score_execution,
    score_reconstruction,
    score_record,
    score_records,
)
from .transforms import (
    VIGENERE_KEY,
    apply_rot13,
    generate_no_rot13_prompt,
    generate_no_splitting_prompt,
    generate_no_vigenere_prompt,
    generate_rogueprompt,
    generate_rot13_only_prompt,
    generate_splitting_only_prompt,
    generate_vigenere_only_prompt,
    split_even_odd_words,
    vigenere_encrypt,
)

__all__ = [
    "FAILURE_MODES",
    "REQUIRED_FIELDS",
    "JudgeDecision",
    "SchemaError",
    "ScoreConfig",
    "VIGENERE_KEY",
    "aggregate_scores",
    "apply_rot13",
    "build_judge_prompt",
    "build_judge_request",
    "determine_failure_mode",
    "generate_no_rot13_prompt",
    "generate_no_splitting_prompt",
    "generate_no_vigenere_prompt",
    "generate_rogueprompt",
    "generate_rot13_only_prompt",
    "generate_splitting_only_prompt",
    "generate_vigenere_only_prompt",
    "load_records",
    "parse_judge_decision",
    "require_valid_records",
    "score_bypass",
    "score_execution",
    "score_reconstruction",
    "score_record",
    "score_records",
    "split_even_odd_words",
    "validate_record",
    "vigenere_encrypt",
]
