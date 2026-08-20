"""Evaluation procedures (Section 4.5), one per hypothesis.

* **H1** -- constraint violation rate, over the manually curated case study.
* **H2** -- Recall@k over synthetic queries.
* **H3** -- per-field precision and recall of constraint extraction.

Descriptive measurements (candidate-set size, relaxation rate, latency) are
also collected, but per Section 2.4 they characterise behaviour and are not
presented as a benefit of the architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean

import pandas as pd

from .config import Config
from .query_parser import QueryParser
from .retrieval import SearchResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# H2: Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(results: list[SearchResult], targets: list[int], k: int) -> float:
    """Proportion of queries whose target appears among the top-k results."""
    if not results:
        return 0.0
    hits = sum(1 for r, t in zip(results, targets) if t in r.indices[:k])
    return hits / len(results)


def reciprocal_rank(result: SearchResult, target: int) -> float:
    """1/rank of the target, or 0 if it is absent from the returned list."""
    if target in result.indices:
        return 1.0 / (result.indices.index(target) + 1)
    return 0.0


# ---------------------------------------------------------------------------
# H1: constraint violation rate
# ---------------------------------------------------------------------------

def constraint_violation_rate(
    result: SearchResult,
    catalog: pd.DataFrame,
    gold_constraints: dict[str, str],
    k: int = 10,
) -> float:
    """Fraction of top-k results whose metadata contradicts a stated constraint.

    This is the metric most directly tied to the argument of this work, and it
    needs no relevance judgment: a violation is decided by the catalog's own
    metadata rather than by opinion.
    """
    if not gold_constraints or not result.indices:
        return 0.0
    top = result.indices[:k]
    violations = 0
    for index in top:
        row = catalog.iloc[index]
        for fld, expected in gold_constraints.items():
            if fld not in catalog.columns:
                continue
            actual = str(row.get(fld, "")).strip().lower()
            if actual and actual != str(expected).strip().lower():
                violations += 1
                break
    return violations / len(top)


# ---------------------------------------------------------------------------
# H3: constraint extraction accuracy
# ---------------------------------------------------------------------------

@dataclass
class ExtractionScore:
    field: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_extraction(
    parser: QueryParser,
    queries: list[str],
    gold: list[dict[str, str]],
) -> dict[str, ExtractionScore]:
    """Compare extracted filters against manually annotated ones, per field.

    A false positive here is more costly than a false negative, since a wrong
    filter removes correct products from consideration entirely; reporting the
    two separately is therefore necessary rather than cosmetic.
    """
    scores: dict[str, ExtractionScore] = {}

    def score_for(fld: str) -> ExtractionScore:
        return scores.setdefault(fld, ExtractionScore(field=fld))

    for query, gold_filters in zip(queries, gold):
        predicted = parser.parse(query).hard_filters
        for fld in set(predicted) | set(gold_filters):
            entry = score_for(fld)
            pred_value = str(predicted.get(fld, "")).strip().lower()
            gold_value = str(gold_filters.get(fld, "")).strip().lower()
            if pred_value and pred_value == gold_value:
                entry.true_positives += 1
            elif pred_value and pred_value != gold_value:
                entry.false_positives += 1
                if gold_value:
                    entry.false_negatives += 1
            elif gold_value:
                entry.false_negatives += 1
    return scores


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class SystemReport:
    """Everything reported for one system over one query set."""

    system: str
    n_queries: int
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_candidates: float = 0.0
    relaxation_rate: float = 0.0
    parse_failure_rate: float = 0.0
    mean_latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["recall"] = {f"recall@{k}": v for k, v in self.recall.items()}
        return payload


def build_report(
    system_name: str,
    results: list[SearchResult],
    targets: list[int],
    cfg: Config,
) -> SystemReport:
    """Aggregate per-query results into the numbers reported in the monograph."""
    if not results:
        return SystemReport(system=system_name, n_queries=0)

    latency_keys = sorted({key for r in results for key in r.timings_ms})
    return SystemReport(
        system=system_name,
        n_queries=len(results),
        recall={k: recall_at_k(results, targets, k) for k in cfg.recall_k_values},
        mrr=mean(reciprocal_rank(r, t) for r, t in zip(results, targets)),
        mean_candidates=mean(r.n_candidates for r in results),
        relaxation_rate=sum(1 for r in results if r.was_relaxed) / len(results),
        parse_failure_rate=sum(1 for r in results if r.parse_failed) / len(results),
        mean_latency_ms={
            key: mean(r.timings_ms.get(key, 0.0) for r in results)
            for key in latency_keys
        },
    )


def save_reports(reports: list[SystemReport], cfg: Config, name: str = "evaluation") -> Path:
    path = cfg.results_dir / f"{name}.json"
    path.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    _save_markdown(reports, cfg, name)
    return path


def _save_markdown(reports: list[SystemReport], cfg: Config, name: str) -> Path:
    k_values = sorted({k for r in reports for k in r.recall})
    header = ["| System | Queries |"] + [f" R@{k} |" for k in k_values]
    lines = [
        "".join(header) + " MRR | Mean candidates | Relaxation rate | Mean latency (ms) |",
        "|---|---|" + "---|" * (len(k_values) + 4),
    ]
    for report in reports:
        cells = [f"| {report.system} | {report.n_queries} |"]
        cells += [f" {report.recall.get(k, 0.0):.3f} |" for k in k_values]
        cells.append(f" {report.mrr:.3f} |")
        cells.append(f" {report.mean_candidates:.0f} |")
        cells.append(f" {report.relaxation_rate:.1%} |")
        cells.append(f" {report.mean_latency_ms.get('total_ms', 0.0):.1f} |")
        lines.append("".join(cells))
    path = cfg.results_dir / f"{name}.md"
    path.write_text("\n".join(lines) + "\n")
    return path
