"""Run the manually curated qualitative case study (Section 4.5, H1).

Reads a CSV of hand-written queries with their annotated hard constraints,
runs both systems over each one, and reports the constraint violation rate
side by side, plus the full result tables for direct inspection.

Create ``data/case_study.csv`` with the columns:

    query,product_type_name,colour_group_name,index_group_name,notes

Leave a constraint cell empty when the query does not state that constraint.
Include cases designed to stress the distinction under study: garment types
with close neighbours in the catalog, and queries stating one colour while
describing a style typical of another.

Usage:
    python scripts/run_case_study.py [--input data/case_study.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fashion_retrieval.config import Config
from fashion_retrieval.data import read_prepared_catalog
from fashion_retrieval.embeddings import build_embedder, load_embeddings
from fashion_retrieval.evaluation import constraint_violation_rate
from fashion_retrieval.label_space import load_vocabulary
from fashion_retrieval.llm import build_llm
from fashion_retrieval.query_parser import QueryParser
from fashion_retrieval.retrieval import BaselineRetriever, HybridRetriever, format_results

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ["article_id", "prod_name", "product_type_name", "colour_group_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "case_study.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--llm", default=None,
        choices=["mock", "ollama", "transformers", "openai-compatible"],
    )
    parser.add_argument("--llm-model", default=None, dest="llm_model")
    parser.add_argument("--embeddings", default=None,
                        choices=["hashing", "sentence-transformers"])
    args = parser.parse_args()

    cfg = Config.from_env()
    if args.llm:
        cfg.llm_backend = args.llm
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.embeddings:
        cfg.embedding_backend = args.embeddings

    if not args.input.exists():
        raise SystemExit(
            f"Case study file not found at {args.input}. See this script's "
            "docstring for the expected columns."
        )

    cases = pd.read_csv(args.input).fillna("")
    catalog = read_prepared_catalog(cfg)
    embeddings = load_embeddings(cfg)
    embedder = build_embedder(cfg)
    vocabulary = load_vocabulary(cfg)

    baseline = BaselineRetriever(catalog, embeddings, embedder)
    hybrid = HybridRetriever(
        catalog, embeddings, embedder,
        QueryParser(build_llm(cfg), vocabulary), cfg.min_candidates,
    )

    rows = []
    for _, case in cases.iterrows():
        query = str(case["query"])
        gold = {
            fld: str(case[fld]).strip()
            for fld in vocabulary
            if fld in cases.columns and str(case[fld]).strip()
        }

        base_result = baseline.search(query, args.k)
        hyb_result = hybrid.search(query, args.k)

        base_violation = constraint_violation_rate(base_result, catalog, gold, args.k)
        hyb_violation = constraint_violation_rate(hyb_result, catalog, gold, args.k)

        print("=" * 78)
        print(f"QUERY: {query}")
        print(f"STATED CONSTRAINTS: {gold or '(none)'}")
        print(f"EXTRACTED FILTERS:  {hyb_result.applied_filters} "
              f"(relaxed: {hyb_result.relaxed_filters or 'none'})")
        print(f"\n-- baseline (violation rate {base_violation:.1%}) --")
        print(format_results(base_result, catalog, DISPLAY).to_string(index=False))
        print(f"\n-- hybrid (violation rate {hyb_violation:.1%}) --")
        print(format_results(hyb_result, catalog, DISPLAY).to_string(index=False))
        print()

        rows.append({
            "query": query,
            "baseline_violation_rate": base_violation,
            "hybrid_violation_rate": hyb_violation,
            "extracted_filters": str(hyb_result.applied_filters),
            "relaxed": str(hyb_result.relaxed_filters),
            "n_candidates": hyb_result.n_candidates,
        })

    summary = pd.DataFrame(rows)
    out = cfg.results_dir / "case_study.csv"
    summary.to_csv(out, index=False)

    print("=" * 78)
    print("SUMMARY")
    print(f"  mean violation rate, baseline: "
          f"{summary['baseline_violation_rate'].mean():.1%}")
    print(f"  mean violation rate, hybrid:   "
          f"{summary['hybrid_violation_rate'].mean():.1%}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
