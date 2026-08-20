"""Baseline and hybrid retrieval systems (Sections 4.3 and 4.4).

Both systems share the same ranking procedure -- cosine similarity between a
query embedding and product embeddings -- and differ only in what happens to
the query before that comparison. Keeping the ranking identical is what makes
the comparison attributable to the query-understanding stage alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .embeddings import Embedder, cosine_scores
from .query_parser import ParsedQuery, QueryParser

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Outcome of a single search, including the diagnostics we report."""

    query: str
    indices: list[int]
    scores: list[float]
    applied_filters: dict[str, str] = field(default_factory=dict)
    relaxed_filters: dict[str, str] = field(default_factory=dict)
    n_candidates: int = 0
    parse_failed: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def was_relaxed(self) -> bool:
        return bool(self.relaxed_filters)


def _top_k(scores: np.ndarray, candidates: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """Return the indices (into the catalog) and scores of the best k candidates."""
    if candidates.size == 0:
        return [], []
    k = min(k, candidates.size)
    subset = scores[candidates]
    order = np.argpartition(-subset, k - 1)[:k]
    order = order[np.argsort(-subset[order])]
    chosen = candidates[order]
    return [int(i) for i in chosen], [float(s) for s in subset[order]]


class BaselineRetriever:
    """Purely embedding-based semantic search over the full catalog."""

    name = "baseline"

    def __init__(self, catalog: pd.DataFrame, embeddings: np.ndarray, embedder: Embedder):
        if len(catalog) != embeddings.shape[0]:
            raise ValueError(
                f"Catalog has {len(catalog)} rows but embedding matrix has "
                f"{embeddings.shape[0]}; they must be aligned."
            )
        self.catalog = catalog
        self.embeddings = embeddings
        self.embedder = embedder
        self._all = np.arange(len(catalog))

    def search(self, query: str, k: int = 10) -> SearchResult:
        t0 = time.perf_counter()
        vector = self.embedder.encode([query])[0]
        t1 = time.perf_counter()
        scores = cosine_scores(vector, self.embeddings)
        indices, top_scores = _top_k(scores, self._all, k)
        t2 = time.perf_counter()
        return SearchResult(
            query=query,
            indices=indices,
            scores=top_scores,
            n_candidates=len(self.catalog),
            timings_ms={
                "embed_ms": (t1 - t0) * 1000,
                "rank_ms": (t2 - t1) * 1000,
                "total_ms": (t2 - t0) * 1000,
            },
        )


class HybridRetriever:
    """LLM-based hard filtering followed by soft-intent ranking."""

    name = "hybrid"

    def __init__(
        self,
        catalog: pd.DataFrame,
        embeddings: np.ndarray,
        embedder: Embedder,
        parser: QueryParser,
        min_candidates: int = 20,
    ):
        if len(catalog) != embeddings.shape[0]:
            raise ValueError("Catalog and embedding matrix are not aligned.")
        self.catalog = catalog
        self.embeddings = embeddings
        self.embedder = embedder
        self.parser = parser
        self.min_candidates = min_candidates
        self._all = np.arange(len(catalog))
        # Pre-lowercased columns make filtering a cheap vectorised comparison.
        self._normalised = {
            fld: catalog[fld].astype(str).str.strip().str.lower()
            for fld in parser.vocabulary
            if fld in catalog.columns
        }

    # -- filtering ---------------------------------------------------------
    def _mask_for(self, filters: dict[str, str]) -> np.ndarray:
        mask = np.ones(len(self.catalog), dtype=bool)
        for fld, value in filters.items():
            column = self._normalised.get(fld)
            if column is None:
                continue
            mask &= (column == str(value).strip().lower()).to_numpy()
        return mask

    def apply_filters(
        self, parsed: ParsedQuery
    ) -> tuple[np.ndarray, dict[str, str], dict[str, str]]:
        """Apply the hard filters, relaxing them if too few candidates survive.

        Filters are dropped one at a time in ascending order of confidence
        until at least ``min_candidates`` products remain. The relaxations
        that occurred are returned so that they can be reported: a system that
        frequently discards its own filters is behaving, in those cases, as
        the baseline does.
        """
        active = dict(parsed.hard_filters)
        relaxed: dict[str, str] = {}
        if not active:
            return self._all, active, relaxed

        mask = self._mask_for(active)
        # ordered_filters() is descending by confidence; drop from the end.
        order = [f for f, _ in parsed.ordered_filters()]
        while mask.sum() < self.min_candidates and order:
            weakest = order.pop()
            relaxed[weakest] = active.pop(weakest)
            log.debug("Relaxing filter %s for query %r", weakest, parsed.raw_query)
            mask = self._mask_for(active) if active else np.ones(len(self.catalog), bool)

        return np.flatnonzero(mask), active, relaxed

    # -- search ------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> SearchResult:
        t0 = time.perf_counter()
        parsed = self.parser.parse(query)
        t1 = time.perf_counter()
        candidates, applied, relaxed = self.apply_filters(parsed)
        t2 = time.perf_counter()
        vector = self.embedder.encode([parsed.ranking_text])[0]
        t3 = time.perf_counter()
        scores = cosine_scores(vector, self.embeddings)
        indices, top_scores = _top_k(scores, candidates, k)
        t4 = time.perf_counter()

        return SearchResult(
            query=query,
            indices=indices,
            scores=top_scores,
            applied_filters=applied,
            relaxed_filters=relaxed,
            n_candidates=int(candidates.size),
            parse_failed=parsed.parse_failed,
            timings_ms={
                "parse_ms": (t1 - t0) * 1000,
                "filter_ms": (t2 - t1) * 1000,
                "embed_ms": (t3 - t2) * 1000,
                "rank_ms": (t4 - t3) * 1000,
                "total_ms": (t4 - t0) * 1000,
            },
        )


def format_results(result: SearchResult, catalog: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Render a search result as a small table for inspection."""
    if not result.indices:
        return pd.DataFrame(columns=columns + ["score"])
    available = [c for c in columns if c in catalog.columns]
    table = catalog.iloc[result.indices][available].copy()
    table["score"] = result.scores
    return table.reset_index(drop=True)
