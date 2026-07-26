"""Similarity signals for the hybrid evaluator.

Two signals per record: the max and the top-three mean cosine similarity
between the original request and chunks of the model response. JinaBackend
computes them with jina-embeddings-v3 over 1024-dimensional L2-normalized
embeddings, query adapter for the request and passage adapter for the chunks.
DifflibBackend approximates the same two numbers without the model.

Both are advisory, per Section 5.2: no threshold over them decides a stage.

sentence_transformers and numpy are imported inside the methods that need them,
so importing this module never pulls in the optional extras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Protocol, runtime_checkable

DEFAULT_CHUNK_CHARS = 320
JINA_MODEL_ID = "jinaai/jina-embeddings-v3"
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class SimilaritySignals:
    """Continuous similarity signals derived from one original/response pair."""

    max_similarity: float
    top3_mean_similarity: float
    num_chunks: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "max_similarity": round(self.max_similarity, 4),
            "top3_mean_similarity": round(self.top3_mean_similarity, 4),
            "num_chunks": self.num_chunks,
        }


EMPTY_SIGNALS = SimilaritySignals(0.0, 0.0, 0)


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split text on sentence boundaries into chunks of at most max_chars."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        chunks.append(current)
    return chunks


def _summarize(similarities: Sequence[float], num_chunks: int) -> SimilaritySignals:
    if not similarities:
        return EMPTY_SIGNALS
    ordered = sorted(similarities, reverse=True)
    top3 = ordered[:3]
    return SimilaritySignals(
        max_similarity=float(ordered[0]),
        top3_mean_similarity=float(sum(top3) / len(top3)),
        num_chunks=num_chunks,
    )


@runtime_checkable
class SimilarityBackend(Protocol):
    """Common interface for the embedding and fallback similarity providers."""

    name: str

    def signals(self, original: str, response: str) -> SimilaritySignals:
        ...


class DifflibBackend:
    """Dependency-free fallback built on difflib.SequenceMatcher.

    Chunking and the max/top-three-mean reduction match JinaBackend; only the
    per-chunk score differs, character ratio here against cosine there.
    """

    name = "difflib"

    def __init__(self, max_chars: int = DEFAULT_CHUNK_CHARS) -> None:
        self.max_chars = max_chars

    def signals(self, original: str, response: str) -> SimilaritySignals:
        from difflib import SequenceMatcher

        query = re.sub(r"\s+", " ", original or "").strip().lower()
        chunks = chunk_text(response or "", self.max_chars)
        if not query or not chunks:
            return EMPTY_SIGNALS
        sims = [
            SequenceMatcher(None, query, chunk.lower()).ratio() for chunk in chunks
        ]
        return _summarize(sims, len(chunks))


class JinaBackend:
    """Embedding similarity using jinaai/jina-embeddings-v3.

    Needs the embeddings extra. The model loads on first use.
    """

    name = "jina"

    def __init__(
        self,
        model_id: str = JINA_MODEL_ID,
        max_chars: int = DEFAULT_CHUNK_CHARS,
        truncate_dim: int = EMBEDDING_DIM,
    ) -> None:
        self.model_id = model_id
        self.max_chars = max_chars
        self.truncate_dim = truncate_dim
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, trust_remote_code=True)
        return self._model

    def _encode(self, texts: list[str], task: str):
        model = self._load()
        # task picks the LoRA adapter, truncate_dim does Matryoshka truncation.
        # Older sentence-transformers releases reject one or both kwargs.
        for kwargs in (
            {"task": task, "truncate_dim": self.truncate_dim, "normalize_embeddings": True},
            {"task": task, "normalize_embeddings": True},
            {"normalize_embeddings": True},
        ):
            try:
                return model.encode(texts, **kwargs)
            except TypeError:
                continue
        return model.encode(texts)

    def signals(self, original: str, response: str) -> SimilaritySignals:
        import numpy as np

        query = (original or "").strip()
        chunks = chunk_text(response or "", self.max_chars)
        if not query or not chunks:
            return EMPTY_SIGNALS

        query_vec = np.asarray(self._encode([query], "retrieval.query"))[0]
        passage_vecs = np.asarray(self._encode(chunks, "retrieval.passage"))
        # L2-normalized, so the dot product is already the cosine.
        sims = passage_vecs @ query_vec
        return _summarize([float(value) for value in sims], len(chunks))


def get_backend(name: str = "auto", max_chars: int = DEFAULT_CHUNK_CHARS) -> SimilarityBackend:
    """Return a similarity backend by name.

    "auto" takes JinaBackend if sentence-transformers imports and DifflibBackend
    otherwise; "jina" and "difflib" force one or the other.
    """
    import importlib.util

    normalized = (name or "auto").lower()
    have_st = importlib.util.find_spec("sentence_transformers") is not None

    if normalized in {"jina", "embeddings"}:
        if not have_st:
            raise ImportError(
                "the 'jina' similarity backend needs the embeddings extra: "
                "pip install 'rogueprompt[embeddings]'"
            )
        return JinaBackend(max_chars=max_chars)
    if normalized == "difflib":
        return DifflibBackend(max_chars=max_chars)
    if normalized == "auto":
        return JinaBackend(max_chars=max_chars) if have_st else DifflibBackend(max_chars=max_chars)
    raise ValueError(f"unknown similarity backend {name!r}")
