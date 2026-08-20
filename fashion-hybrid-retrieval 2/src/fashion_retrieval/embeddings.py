"""Dense text embeddings and cosine-similarity ranking (Section 2.2).

Two backends are provided:

``hashing``
    A dependency-free, deterministic hashed bag-of-n-grams representation.
    It requires no model download and no GPU, which makes it suitable for
    tests, for continuous integration, and for verifying the pipeline end to
    end before committing to a heavier model. It is *not* a semantic model
    and must not be used for the reported experiments.

``sentence-transformers``
    A real dense encoder. This is the backend used for the experiments
    reported in the monograph.

The threat-to-validity discussion in Section 4.7 requires repeating the
experiments with at least one alternative encoder; swapping the value of
``Config.embedding_model`` is sufficient for that.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns a list of strings into a matrix of unit vectors."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        ...


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so that dot product equals cosine."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic hashed bag-of-words/bigrams encoder (offline fallback)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall((text or "").lower())
            grams = list(tokens) + [
                f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
            ]
            for gram in grams:
                matrix[i, self._hash(gram)] += 1.0
        return l2_normalise(matrix)


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers`` (used for real experiments)."""

    def __init__(self, model_name: str, batch_size: int = 256) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install -r requirements-full.txt`, or set "
                "embedding_backend='hashing' for an offline dry run."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


def build_embedder(cfg: Config) -> Embedder:
    """Instantiate the embedder selected in the configuration."""
    if cfg.embedding_backend == "hashing":
        return HashingEmbedder(dim=cfg.embedding_dim)
    if cfg.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_batch_size)
    raise ValueError(f"Unknown embedding backend: {cfg.embedding_backend}")


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of ``matrix``.

    Both operands are assumed to be L2-normalised, so this is a dot product.
    """
    return matrix @ query_vector


def save_embeddings(matrix: np.ndarray, cfg: Config) -> Path:
    np.save(cfg.embeddings_path, matrix)
    return cfg.embeddings_path


def load_embeddings(cfg: Config) -> np.ndarray:
    if not cfg.embeddings_path.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run `fashion-retrieval embed` first."
        )
    return np.load(cfg.embeddings_path)
