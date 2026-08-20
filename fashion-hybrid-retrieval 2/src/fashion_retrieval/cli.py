"""Command-line interface for the whole pipeline.

Typical run, entirely offline:

    python -m fashion_retrieval prepare
    python -m fashion_retrieval label-space
    python -m fashion_retrieval embed
    python -m fashion_retrieval gen-queries
    python -m fashion_retrieval evaluate
    python -m fashion_retrieval search "a black midi dress for a wedding"

Add ``--llm ollama --embeddings sentence-transformers`` to run the real
configuration used for the reported experiments. Both are free and local: no
API key, no account, no billing.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import Config
from .data import PRODUCT_TEXT_COLUMN, load_catalog, read_prepared_catalog, save_catalog
from .embeddings import build_embedder, load_embeddings, save_embeddings
from .evaluation import build_report, save_reports
from .label_space import (
    analyse_label_space,
    build_vocabulary,
    load_vocabulary,
    save_report,
    save_vocabulary,
    stats_to_markdown,
)
from .llm import build_llm
from .query_parser import QueryParser
from .retrieval import BaselineRetriever, HybridRetriever, format_results
from .synthetic_queries import (
    export_plausibility_sample,
    generate_queries,
    load_queries,
    save_queries,
)

DISPLAY_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "index_group_name",
]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if getattr(args, "embeddings", None):
        cfg.embedding_backend = args.embeddings
    if getattr(args, "llm", None):
        cfg.llm_backend = args.llm
    if getattr(args, "llm_model", None):
        cfg.llm_model = args.llm_model
    if getattr(args, "llm_host", None):
        cfg.llm_host = args.llm_host
    if getattr(args, "max_articles", None):
        cfg.max_articles = args.max_articles
    if getattr(args, "top_k", None):
        cfg.top_k = args.top_k
    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = load_catalog(cfg)
    path = save_catalog(catalog, cfg)
    print(f"Prepared {len(catalog)} articles -> {path}")
    return 0


def cmd_label_space(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = read_prepared_catalog(cfg)
    stats = analyse_label_space(catalog, cfg)
    report_path = save_report(stats, cfg)

    usable = [s.field for s in stats if s.usable_as_filter]
    chosen = [f for f in cfg.filter_fields if f in usable] or usable[:3]
    vocabulary = build_vocabulary(catalog, cfg, chosen)
    vocab_path = save_vocabulary(vocabulary, cfg)

    print(stats_to_markdown(stats))
    print(f"\nReport -> {report_path}")
    print(f"Filter vocabulary ({', '.join(chosen)}) -> {vocab_path}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = read_prepared_catalog(cfg)
    embedder = build_embedder(cfg)
    matrix = embedder.encode(catalog[PRODUCT_TEXT_COLUMN].fillna("").tolist())
    path = save_embeddings(matrix, cfg)
    print(f"Encoded {matrix.shape[0]} products into {matrix.shape[1]} dims "
          f"with '{embedder.name}' -> {path}")
    return 0


def cmd_gen_queries(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = read_prepared_catalog(cfg)
    llm = build_llm(cfg)
    queries = generate_queries(catalog, llm, cfg, n=args.n)
    path = save_queries(queries, cfg)
    sample = export_plausibility_sample(queries, catalog, cfg)
    print(f"Generated {len(queries)} queries -> {path}")
    print(f"Plausibility sample for manual annotation -> {sample}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = read_prepared_catalog(cfg)
    embeddings = load_embeddings(cfg)
    embedder = build_embedder(cfg)
    vocabulary = load_vocabulary(cfg)
    parser = QueryParser(build_llm(cfg), vocabulary)

    baseline = BaselineRetriever(catalog, embeddings, embedder)
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, cfg.min_candidates)

    queries = load_queries(cfg)
    targets = [q.target_index for q in queries]

    reports = []
    for system in (baseline, hybrid):
        results = [system.search(q.query, k=max(cfg.recall_k_values)) for q in queries]
        reports.append(build_report(system.name, results, targets, cfg))

    path = save_reports(reports, cfg)
    for report in reports:
        recall = " ".join(f"R@{k}={v:.3f}" for k, v in report.recall.items())
        print(f"{report.system:<9} {recall} MRR={report.mrr:.3f} "
              f"cand={report.mean_candidates:.0f} relax={report.relaxation_rate:.1%}")
    print(f"\nFull report -> {path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    catalog = read_prepared_catalog(cfg)
    embeddings = load_embeddings(cfg)
    embedder = build_embedder(cfg)

    baseline = BaselineRetriever(catalog, embeddings, embedder)
    print(f"\n=== baseline === {args.query!r}")
    print(format_results(baseline.search(args.query, cfg.top_k), catalog, DISPLAY_COLUMNS)
          .to_string(index=False))

    parser = QueryParser(build_llm(cfg), load_vocabulary(cfg))
    hybrid = HybridRetriever(catalog, embeddings, embedder, parser, cfg.min_candidates)
    result = hybrid.search(args.query, cfg.top_k)
    print(f"\n=== hybrid === filters={result.applied_filters} "
          f"relaxed={result.relaxed_filters} candidates={result.n_candidates}")
    print(format_results(result, catalog, DISPLAY_COLUMNS).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Options accepted both before and after the subcommand."""
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--embeddings", choices=["hashing", "sentence-transformers"])
    parser.add_argument(
        "--llm",
        choices=["mock", "ollama", "transformers", "openai-compatible"],
        help="LLM backend (all free and local; see README)",
    )
    parser.add_argument("--llm-model", dest="llm_model",
                        help="e.g. qwen2.5:3b-instruct for the ollama backend")
    parser.add_argument("--llm-host", dest="llm_host",
                        help="base URL for ollama / openai-compatible backends")
    parser.add_argument("--max-articles", type=int, dest="max_articles")


def build_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common)

    parser = argparse.ArgumentParser(
        prog="fashion-retrieval",
        description="Hybrid retrieval for fashion e-commerce search.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", parents=[common],
                   help="load and clean the catalog").set_defaults(func=cmd_prepare)
    sub.add_parser("label-space", parents=[common],
                   help="analyse fields and build the vocabulary").set_defaults(
        func=cmd_label_space
    )
    sub.add_parser("embed", parents=[common],
                   help="encode product texts").set_defaults(func=cmd_embed)

    gen = sub.add_parser("gen-queries", parents=[common],
                         help="generate synthetic evaluation queries")
    gen.add_argument("-n", type=int, default=None, help="number of queries")
    gen.set_defaults(func=cmd_gen_queries)

    sub.add_parser("evaluate", parents=[common],
                   help="run both systems and report metrics").set_defaults(
        func=cmd_evaluate
    )

    search = sub.add_parser("search", parents=[common],
                            help="run a single query through both systems")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, dest="top_k")
    search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)
    pd.set_option("display.max_colwidth", 40)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
