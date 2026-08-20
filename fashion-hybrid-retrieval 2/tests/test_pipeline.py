"""Tests for the retrieval pipeline.

Everything here runs offline: the hashing embedder needs no download and the
mock LLM needs no API key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion_retrieval.config import Config
from fashion_retrieval.data import build_product_text
from fashion_retrieval.embeddings import HashingEmbedder, cosine_scores
from fashion_retrieval.evaluation import (
    build_report,
    constraint_violation_rate,
    evaluate_extraction,
    recall_at_k,
)
from fashion_retrieval.label_space import analyse_field, build_vocabulary
from fashion_retrieval.llm import MockLLM, extract_json
from fashion_retrieval.query_parser import QueryParser
from fashion_retrieval.retrieval import BaselineRetriever, HybridRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog() -> pd.DataFrame:
    rows = []
    for i in range(60):
        product_type = ["Dress", "Skirt", "Trousers"][i % 3]
        colour = ["Black", "Red", "White"][i % 3]
        audience = ["Ladieswear", "Menswear"][i % 2]
        rows.append(
            {
                "article_id": f"{i:09d}",
                "prod_name": f"{colour} {product_type}",
                "product_type_name": product_type,
                "product_group_name": "Garment Full body",
                "colour_group_name": colour,
                "graphical_appearance_name": "Solid",
                "garment_group_name": "Jersey Basic",
                "index_group_name": audience,
                "detail_desc": (
                    f"Relaxed {product_type.lower()} in soft jersey with a round "
                    f"neckline. Midi length. Unlined."
                ),
            }
        )
    df = pd.DataFrame(rows)
    df["product_text"] = df.apply(
        lambda r: build_product_text(
            r, ["product_type_name", "colour_group_name", "index_group_name"]
        ),
        axis=1,
    )
    return df


@pytest.fixture
def vocabulary(catalog) -> dict[str, list[str]]:
    cfg = Config(min_value_frequency=1)
    return build_vocabulary(catalog, cfg)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def embeddings(catalog, embedder) -> np.ndarray:
    return embedder.encode(catalog["product_text"].tolist())


@pytest.fixture
def parser(vocabulary) -> QueryParser:
    return QueryParser(MockLLM(), vocabulary)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_embeddings_are_unit_norm(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic(embedder):
    a = embedder.encode(["a black midi dress"])
    b = embedder.encode(["a black midi dress"])
    assert np.array_equal(a, b)


def test_cosine_scores_are_bounded(embeddings, embedder):
    scores = cosine_scores(embedder.encode(["black dress"])[0], embeddings)
    assert scores.shape == (len(embeddings),)
    assert scores.min() >= -1.0001 and scores.max() <= 1.0001


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

def test_analyse_field_flags_high_cardinality(catalog):
    cfg = Config(max_filter_cardinality=1, min_value_frequency=1)
    stats = analyse_field(catalog, "product_type_name", cfg)
    assert stats.n_distinct == 3
    assert not stats.usable_as_filter


def test_vocabulary_excludes_rare_values(catalog):
    catalog = pd.concat(
        [catalog, catalog.head(1).assign(colour_group_name="Chartreuse")],
        ignore_index=True,
    )
    vocab = build_vocabulary(catalog, Config(min_value_frequency=5))
    assert "Chartreuse" not in vocab["colour_group_name"]
    assert "Black" in vocab["colour_group_name"]


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert extract_json("not json at all") == {}


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

def test_parser_extracts_known_values(parser):
    parsed = parser.parse("a black dress for ladieswear")
    assert parsed.hard_filters["colour_group_name"] == "Black"
    assert parsed.hard_filters["product_type_name"] == "Dress"


def test_parser_drops_values_outside_the_vocabulary(vocabulary):
    class BadLLM:
        name = "bad"

        def complete(self, system, user):
            return (
                '{"hard_filters": {"colour_group_name": "Chartreuse", '
                '"made_up_field": "x"}, "soft_intent": "flowy", '
                '"confidence": {"colour_group_name": 0.9}}'
            )

    parsed = QueryParser(BadLLM(), vocabulary).parse("a chartreuse dress")
    assert parsed.hard_filters == {}
    assert "colour_group_name" in parsed.dropped_values
    assert "made_up_field" in parsed.dropped_values


def test_parser_marks_failure_on_unparseable_output(vocabulary):
    class SilentLLM:
        name = "silent"

        def complete(self, system, user):
            return "I'm afraid I can't do that."

    parsed = QueryParser(SilentLLM(), vocabulary).parse("a red skirt")
    assert parsed.parse_failed
    assert parsed.ranking_text == "a red skirt"


def test_ranking_text_falls_back_to_raw_query(parser):
    parsed = parser.parse("dress")
    assert parsed.ranking_text.strip()


def test_filters_are_ordered_by_confidence(vocabulary):
    class ConfidentLLM:
        name = "confident"

        def complete(self, system, user):
            return (
                '{"hard_filters": {"colour_group_name": "Black", '
                '"product_type_name": "Dress"}, "soft_intent": "flowy", '
                '"confidence": {"colour_group_name": 0.4, "product_type_name": 0.95}}'
            )

    parsed = QueryParser(ConfidentLLM(), vocabulary).parse("q")
    assert [f for f, _ in parsed.ordered_filters()][0] == "product_type_name"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_baseline_returns_k_results(catalog, embeddings, embedder):
    result = BaselineRetriever(catalog, embeddings, embedder).search("black dress", k=5)
    assert len(result.indices) == 5
    assert result.scores == sorted(result.scores, reverse=True)
    assert result.n_candidates == len(catalog)


def test_baseline_rejects_misaligned_embeddings(catalog, embeddings, embedder):
    with pytest.raises(ValueError):
        BaselineRetriever(catalog, embeddings[:-1], embedder)


def test_hybrid_respects_hard_filters(catalog, embeddings, embedder, parser):
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, min_candidates=1)
    result = hybrid.search("a black dress", k=10)
    assert result.applied_filters.get("product_type_name") == "Dress"
    retrieved = catalog.iloc[result.indices]
    assert set(retrieved["product_type_name"]) == {"Dress"}
    assert set(retrieved["colour_group_name"]) == {"Black"}


def test_hybrid_relaxes_filters_when_candidate_set_is_too_small(
    catalog, embeddings, embedder, parser
):
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, min_candidates=50)
    result = hybrid.search("a black dress", k=10)
    assert result.was_relaxed
    assert result.n_candidates >= 50


def test_hybrid_without_filters_behaves_like_baseline(
    catalog, embeddings, embedder, vocabulary
):
    class EmptyLLM:
        name = "empty"

        def complete(self, system, user):
            return '{"hard_filters": {}, "soft_intent": "something cosy"}'

    hybrid = HybridRetriever(
        catalog, embeddings, embedder, QueryParser(EmptyLLM(), vocabulary)
    )
    baseline = BaselineRetriever(catalog, embeddings, embedder)
    hybrid_result = hybrid.search("something cosy", k=5)
    baseline_result = baseline.search("something cosy", k=5)
    assert hybrid_result.indices == baseline_result.indices


def test_search_records_timings(catalog, embeddings, embedder, parser):
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, min_candidates=1)
    result = hybrid.search("a red skirt", k=3)
    assert {"parse_ms", "filter_ms", "embed_ms", "rank_ms", "total_ms"} <= set(
        result.timings_ms
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_recall_at_k(catalog, embeddings, embedder):
    baseline = BaselineRetriever(catalog, embeddings, embedder)
    results = [baseline.search(catalog.iloc[i]["product_text"], k=10) for i in range(5)]
    assert recall_at_k(results, list(range(5)), 10) > 0.0
    assert recall_at_k(results, [999] * 5, 10) == 0.0


def test_constraint_violation_rate(catalog, embeddings, embedder, parser):
    baseline = BaselineRetriever(catalog, embeddings, embedder)
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, min_candidates=1)
    gold = {"product_type_name": "Dress", "colour_group_name": "Black"}
    base = constraint_violation_rate(baseline.search("a black dress", 10), catalog, gold)
    hyb = constraint_violation_rate(hybrid.search("a black dress", 10), catalog, gold)
    assert hyb == 0.0
    assert hyb <= base


def test_evaluate_extraction_counts_errors(vocabulary):
    parser = QueryParser(MockLLM(), vocabulary)
    scores = evaluate_extraction(
        parser,
        ["a black dress", "a red skirt"],
        [
            {"product_type_name": "Dress", "colour_group_name": "Black"},
            {"product_type_name": "Skirt", "colour_group_name": "White"},
        ],
    )
    colour = scores["colour_group_name"]
    assert colour.true_positives == 1
    assert colour.false_positives == 1
    assert 0.0 <= colour.f1 <= 1.0


def test_build_report_aggregates(catalog, embeddings, embedder, tmp_path):
    cfg = Config(processed_dir=tmp_path, results_dir=tmp_path, recall_k_values=(1, 5))
    baseline = BaselineRetriever(catalog, embeddings, embedder)
    results = [baseline.search(catalog.iloc[i]["product_text"], k=5) for i in range(4)]
    report = build_report("baseline", results, list(range(4)), cfg)
    assert report.n_queries == 4
    assert set(report.recall) == {1, 5}
    assert report.relaxation_rate == 0.0
    assert report.mean_latency_ms["total_ms"] >= 0.0
