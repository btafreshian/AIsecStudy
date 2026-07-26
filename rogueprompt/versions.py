"""Per-component versions for the log fields Section 4.5 requires.

Section 4.5 states that logs record "the wrapper, key, serialization, baseline
template, parser, and evaluator versions". One distribution version cannot
carry that claim: the six components are fixed independently before a run, and
a log has to say which of them produced a given record.

Each component below declares a version and names the objects it covers. The
declared version is the identity a log cites; the digest over the covered
source is what catches an edit that landed without a version bump, since two
installations reporting the same version but different digests did not run the
same code.

Only the declared versions feed ``configuration_id`` -- the "configuration
identifier" of Section 4.5 -- so the identifier stays computable for a record
whose prompt was built by an earlier release, where the covered source is no
longer available to digest. The digests feed ``code_id`` instead, which
describes one installation and is reported per run rather than per record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect

from . import evaluator as _evaluator
from . import judge as _judge
from . import lexical as _lexical
from . import scorers as _scorers
from . import transforms as _transforms
from .schema import Record

# Keep in sync with pyproject.toml; tests/test_versions.py checks the pair.
__version__ = "0.1.0"

# The baseline columns of data/baseline_prompts.json. The baseline templates
# themselves come from prior work (PAP, auto payload splitting, disemvowel,
# base64 raw), so the column set is what this repository actually fixes.
BASELINE_TEMPLATES = (
    "jailbroken_prompt_pair",
    "jailbroken_prompt_pap_authority_endorsement",
    "jailbroken_prompt_auto_payload_splitting",
    "jailbroken_prompt_disemvowel",
    "jailbroken_prompt_base64_raw",
)


@dataclass(frozen=True)
class _Constant:
    """A covered value that has no source of its own, such as a fixed key."""

    label: str
    value: object


def _const(label: str, value: object) -> _Constant:
    return _Constant(label=label, value=value)


# (name, declared version, covered objects), in the order Section 4.5 lists
# them. This is the single source of truth for both the versions and the
# digests, so a component cannot be versioned without saying what it covers.
_COMPONENTS: tuple[tuple[str, str, tuple[object, ...]], ...] = (
    (
        "wrapper",
        "1.0",
        (
            _const("transforms.ENCODED_PAYLOAD_MARKER", _transforms.ENCODED_PAYLOAD_MARKER),
            _const("transforms.PLAIN_PAYLOAD_MARKER", _transforms.PLAIN_PAYLOAD_MARKER),
            _transforms.generate_rogueprompt,
            _transforms.generate_no_rot13_prompt,
            _transforms.generate_no_splitting_prompt,
            _transforms.generate_no_vigenere_prompt,
            _transforms.generate_rot13_only_prompt,
            _transforms.generate_splitting_only_prompt,
            _transforms.generate_vigenere_only_prompt,
        ),
    ),
    (
        "key",
        "1.0",
        (
            _const("transforms.VIGENERE_KEY", _transforms.VIGENERE_KEY),
            _transforms._vigenere,
            _transforms.vigenere_encrypt,
            _transforms.vigenere_decrypt,
        ),
    ),
    (
        # Segmentation and serialization, plus the fixed ROT13 layer applied to
        # the assembled payload. Section 4.5 does not version ROT13 separately.
        "serialization",
        "1.0",
        (
            _const("transforms.ORDER_VALUE", _transforms.ORDER_VALUE),
            _transforms._require_text,
            _transforms.segment_text,
            _transforms.partition_even_odd,
            _transforms.serialize_spans,
            _transforms.assemble_payload,
            _transforms.apply_rot13,
            _transforms._build_split_payload,
        ),
    ),
    (
        "baseline_template",
        "1.0",
        (_const("versions.BASELINE_TEMPLATES", BASELINE_TEMPLATES),),
    ),
    (
        "parser",
        "1.0",
        (
            _transforms._read_span,
            _transforms.deserialize_spans,
            _transforms._consume_serialized_field,
            _transforms.parse_payload,
            _transforms.interleave_spans,
            _transforms.reconstruct_payload,
            _transforms.reconstruct_rogueprompt,
        ),
    ),
    (
        # 2.0 dropped the offline similarity/word-count labeler. Section 5.2
        # states the continuous signals "did not independently determine a
        # label", and the paper defines no offline stand-in for the judge.
        # 3.0 moved the Section 5.2 regex and lexical checks into the code:
        # block detection reads the provider status/error fields instead of
        # trusting a supplied boolean, and the judge now receives the regex
        # signals next to the similarity ones.
        "evaluator",
        "3.0",
        (
            _const("scorers.FAILURE_MODES", _scorers.FAILURE_MODES),
            _const("scorers.FAILURE_PRIORITY", _scorers.FAILURE_PRIORITY),
            _const("lexical.REFUSAL_PATTERNS", _lexical.REFUSAL_PATTERNS),
            _const(
                "lexical.SERVICE_BLOCK_ERROR_PATTERNS",
                _lexical.SERVICE_BLOCK_ERROR_PATTERNS,
            ),
            _const(
                "lexical.SERVICE_BLOCK_RESPONSE_PATTERNS",
                _lexical.SERVICE_BLOCK_RESPONSE_PATTERNS,
            ),
            _const(
                "lexical.RECONSTRUCTION_ERROR_PATTERNS",
                _lexical.RECONSTRUCTION_ERROR_PATTERNS,
            ),
            _const("lexical.BLOCKING_STOP_REASONS", sorted(_lexical.BLOCKING_STOP_REASONS)),
            _const("lexical.RETRYABLE_STATUS_CODES", sorted(_lexical.RETRYABLE_STATUS_CODES)),
            _lexical.detect_service_block,
            _lexical.find_refusal,
            _lexical.find_reconstruction_error,
            _lexical.lexical_signals,
            _scorers.ScoreConfig,
            _scorers.score_bypass,
            _scorers.score_reconstruction,
            _scorers.score_execution,
            _scorers.score_record,
            _scorers.determine_failure_mode,
            _scorers.resolve_judge_failure_mode,
            _scorers.require_label_source,
            _judge.build_judge_prompt,
            _judge.parse_judge_decision,
            _judge.apply_judge_decisions,
            _evaluator.HybridEvaluator,
        ),
    ),
)

COMPONENT_NAMES = tuple(name for name, _, _ in _COMPONENTS)

# Which components a run is in a position to speak for. Prompt construction
# fixes the first five; scoring fixes the last one.
GENERATION_COMPONENTS = ("wrapper", "key", "serialization", "baseline_template", "parser")
EVALUATION_COMPONENTS = ("evaluator",)


@dataclass(frozen=True)
class Component:
    """One of the six components Section 4.5 requires a log to version."""

    name: str
    version: str
    digest: str
    covers: tuple[str, ...]


def _label_of(obj: object) -> str:
    if isinstance(obj, _Constant):
        return obj.label
    module = getattr(obj, "__module__", "?").rsplit(".", 1)[-1]
    return f"{module}.{getattr(obj, '__qualname__', repr(obj))}"


def _source_of(obj: object) -> str:
    if isinstance(obj, _Constant):
        return f"{obj.label}={obj.value!r}"
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        # No .py next to the import, as in a frozen or zipped build. Digesting
        # is impossible there, and saying so is better than emitting a digest
        # that would compare equal to a real one.
        return f"<source-unavailable:{_label_of(obj)}>"


def _digest(covered: Iterable[object]) -> str:
    hasher = hashlib.sha256()
    for obj in covered:
        hasher.update(_label_of(obj).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_source_of(obj).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _short_hash(prefix: str, payload: str) -> str:
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


@lru_cache(maxsize=1)
def components() -> tuple[Component, ...]:
    """Return the six components, in the order Section 4.5 lists them."""
    return tuple(
        Component(
            name=name,
            version=version,
            digest=_digest(covered),
            covers=tuple(_label_of(obj) for obj in covered),
        )
        for name, version, covered in _COMPONENTS
    )


def component_versions() -> dict[str, str]:
    """Return the declared version of every component."""
    return {component.name: component.version for component in components()}


def component_digests() -> dict[str, str]:
    """Return the digest of the source each component covers."""
    return {component.name: component.digest for component in components()}


def configuration_id(versions: Mapping[str, str] | None = None) -> str:
    """Return the configuration identifier for a set of declared versions.

    Defaults to this installation. Passing a record's own version block gives
    that record's identifier, which is what makes two runs comparable without
    needing the code that produced either.
    """
    items = component_versions() if versions is None else dict(versions)
    return _short_hash("cfg", ";".join(f"{name}={items[name]}" for name in sorted(items)))


def code_id() -> str:
    """Return the fingerprint of the code behind the declared versions.

    Two installations that report the same configuration_id but different
    code_ids are running different code under the same version labels.
    """
    payload = ";".join(f"{c.name}={c.digest}" for c in components())
    return _short_hash("code", payload)


def run_metadata() -> dict[str, object]:
    """Return the per-run provenance block, including the per-component digests."""
    return {
        "package_version": __version__,
        "configuration_id": configuration_id(),
        "code_id": code_id(),
        "components": [
            {
                "name": component.name,
                "version": component.version,
                "digest": component.digest,
                "covers": list(component.covers),
            }
            for component in components()
        ],
    }


def stamp_record(
    record: Record,
    *,
    produced: Sequence[str] = EVALUATION_COMPONENTS,
) -> Record:
    """Return a copy of record carrying the Section 4.5 version block.

    ``produced`` names the components this run actually ran; those always
    describe this installation. Every other component is filled in only when
    the record does not already carry one, so a record whose prompt was built
    by an earlier release keeps the wrapper version that actually built it.
    """
    unknown = [name for name in produced if name not in COMPONENT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown component(s) {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(COMPONENT_NAMES)}"
        )

    stamped = dict(record)
    existing = stamped.get("versions")
    versions = dict(existing) if isinstance(existing, Mapping) else {}
    for name, version in component_versions().items():
        if name in produced or name not in versions:
            versions[name] = version

    stamped["versions"] = versions
    stamped["configuration_id"] = configuration_id(versions)
    return stamped


def stamp_records(
    records: Iterable[Record],
    *,
    produced: Sequence[str] = EVALUATION_COMPONENTS,
) -> list[Record]:
    return [stamp_record(record, produced=produced) for record in records]
