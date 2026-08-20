"""Synthetic query generation (Section 4.5, quantitative evaluation).

Three controls are built into this module, because the naive version of this
procedure is biased in favour of the hybrid system:

1. **Metadata is withheld from the generator.** The model sees only the
   product name and its free-text description -- never the structured fields
   used as filters, and never the filter schema. Otherwise the generated
   queries would be phrased in the exact vocabulary the filters match
   against, and the hybrid system would be rewarded for a correspondence
   created by the experimental setup rather than by the architecture.
2. **Shopper register is requested, not paraphrase.** The prompt asks for
   underspecified and stylistic queries, so that the query set is not
   restricted to the easy case where query and indexed text are near
   duplicates.
3. **Plausibility is measured, not assumed.** A random sample is exported for
   manual inspection, and the proportion judged plausible is reported.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .config import DESCRIPTION_COLUMN, ID_COLUMN, NAME_COLUMN, Config
from .llm import LLMClient, extract_json

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """SYNTHETIC QUERY GENERATION

You write realistic search queries for a fashion e-commerce website.

You will be shown one product. Write the query a shopper might type into the \
search box if they were hoping to find this product, in the STYLE indicated.

Rules:
- Write as a shopper would: lowercase, short, no punctuation at the end.
- Do NOT copy the product description sentence by sentence. A shopper does not \
know how the catalog words things.
- Do NOT list every attribute. Real queries are underspecified.
- Never mention brand names or product codes.
- Respond with a single JSON object and nothing else: {"query": "..."}
"""

#: Query registers requested from the generator, so the resulting set spans
#: more than one difficulty level.
QUERY_STYLES = [
    "a short, literal query naming the garment and one or two obvious properties",
    "a query that states one hard constraint and otherwise describes a vibe or style",
    "an occasion-driven query, mentioning where or when the shopper would wear it",
    "a vague, underspecified query of three or four words",
]


@dataclass
class SyntheticQuery:
    """A generated query and the product it was generated from."""

    query: str
    target_index: int
    target_id: str
    style: str


def _user_prompt(row: pd.Series, style: str) -> str:
    """Build the generator prompt. Deliberately excludes metadata fields."""
    return (
        f"PRODUCT NAME: {row.get(NAME_COLUMN, '')}\n"
        f"PRODUCT DESCRIPTION: {row.get(DESCRIPTION_COLUMN, '')}\n\n"
        f"STYLE: {style}"
    )


def generate_queries(
    catalog: pd.DataFrame,
    llm: LLMClient,
    cfg: Config,
    n: int | None = None,
) -> list[SyntheticQuery]:
    """Generate one query per sampled product, cycling through the styles."""
    n = n if n is not None else cfg.n_synthetic_queries
    n = min(n, len(catalog))
    rng = random.Random(cfg.random_seed)
    positions = rng.sample(range(len(catalog)), n)

    queries: list[SyntheticQuery] = []
    for i, position in enumerate(positions):
        row = catalog.iloc[position]
        style = QUERY_STYLES[i % len(QUERY_STYLES)]
        raw = llm.complete(SYSTEM_PROMPT, _user_prompt(row, style))
        payload = extract_json(raw)
        text = str(payload.get("query", "")).strip()
        if not text:
            log.warning("Empty query generated for position %d, skipping", position)
            continue
        queries.append(
            SyntheticQuery(
                query=text,
                target_index=int(position),
                target_id=str(row.get(ID_COLUMN, position)),
                style=style,
            )
        )
    log.info("Generated %d synthetic queries", len(queries))
    return queries


def save_queries(queries: list[SyntheticQuery], cfg: Config) -> Path:
    cfg.synthetic_queries_path.write_text(
        json.dumps([asdict(q) for q in queries], indent=2, ensure_ascii=False)
    )
    return cfg.synthetic_queries_path


def load_queries(cfg: Config) -> list[SyntheticQuery]:
    if not cfg.synthetic_queries_path.exists():
        raise FileNotFoundError(
            "Synthetic queries not found. Run `fashion-retrieval gen-queries` first."
        )
    payload = json.loads(cfg.synthetic_queries_path.read_text())
    return [SyntheticQuery(**item) for item in payload]


def export_plausibility_sample(
    queries: list[SyntheticQuery], catalog: pd.DataFrame, cfg: Config
) -> Path:
    """Export a random sample for manual plausibility annotation (control 3).

    The resulting CSV has an empty ``plausible`` column to be filled in by
    hand with 1 or 0; the proportion is reported in the monograph.
    """
    rng = random.Random(cfg.random_seed)
    sample = rng.sample(queries, min(cfg.plausibility_sample_size, len(queries)))
    rows = []
    for item in sample:
        product = catalog.iloc[item.target_index]
        rows.append(
            {
                "query": item.query,
                "style": item.style,
                "target_id": item.target_id,
                "target_name": product.get(NAME_COLUMN, ""),
                "target_description": product.get(DESCRIPTION_COLUMN, ""),
                "plausible": "",
                "notes": "",
            }
        )
    path = cfg.results_dir / "plausibility_sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
