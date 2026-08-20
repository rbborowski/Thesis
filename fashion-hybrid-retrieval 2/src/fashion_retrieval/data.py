"""Catalog loading and preparation (Section 4.2 of the monograph).

The catalog is built from the ``articles.csv`` table of the H&M Personalized
Fashion Recommendations dataset. Only article metadata and descriptions are
used; the transaction and customer tables are not, since this work does not
model user history.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import (
    DESCRIPTION_COLUMN,
    ID_COLUMN,
    NAME_COLUMN,
    TEXT_CONTEXT_FIELDS,
    Config,
)

log = logging.getLogger(__name__)

#: Name of the column holding the text that is embedded for ranking.
PRODUCT_TEXT_COLUMN = "product_text"


def build_product_text(row: pd.Series, context_fields: list[str]) -> str:
    """Assemble the text that represents a product in the embedding space.

    The structured metadata fields are appended to the free-text description
    on purpose: the baseline must have access to the same information the
    hybrid system uses as filters, otherwise any measured difference would
    partly reflect an information asymmetry rather than the architectural
    difference under study (see Section 4.3).
    """
    parts: list[str] = []
    name = row.get(NAME_COLUMN)
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())

    for field_name in context_fields:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    description = row.get(DESCRIPTION_COLUMN)
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())

    return ". ".join(parts)


def load_catalog(cfg: Config, path: Path | None = None) -> pd.DataFrame:
    """Load, clean and prepare the article catalog.

    Steps applied, in order:

    1. read ``articles.csv``;
    2. drop articles without a usable free-text description, since neither
       system can rank them;
    3. build the ``product_text`` column used for embedding;
    4. optionally subsample the catalog for quick local runs.

    Near-duplicate articles differing only in colorway are deliberately kept:
    distinguishing between them is exactly the case where a colour filter is
    expected to matter.
    """
    source = Path(path) if path is not None else cfg.articles_path
    if not source.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {source}. Download articles.csv from "
            "the H&M Kaggle competition into data/raw/, or run "
            "`python scripts/make_sample_catalog.py` to generate a small "
            "synthetic catalog with the same schema for testing."
        )

    log.info("Reading catalog from %s", source)
    df = pd.read_csv(source, dtype={ID_COLUMN: str})

    missing = [c for c in (ID_COLUMN, NAME_COLUMN, DESCRIPTION_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    n_before = len(df)
    df = df[df[DESCRIPTION_COLUMN].notna()].copy()
    df = df[df[DESCRIPTION_COLUMN].str.len() >= cfg.min_description_chars]
    log.info("Dropped %d articles without a usable description", n_before - len(df))

    context_fields = [f for f in TEXT_CONTEXT_FIELDS if f in df.columns]
    df[PRODUCT_TEXT_COLUMN] = df.apply(
        lambda row: build_product_text(row, context_fields), axis=1
    )

    if cfg.max_articles is not None and len(df) > cfg.max_articles:
        df = df.sample(n=cfg.max_articles, random_state=cfg.random_seed)
        log.info("Subsampled catalog to %d articles", len(df))

    df = df.reset_index(drop=True)
    log.info("Prepared catalog with %d articles", len(df))
    return df


def save_catalog(df: pd.DataFrame, cfg: Config) -> Path:
    """Persist the prepared catalog, falling back to CSV if parquet is absent."""
    try:
        df.to_parquet(cfg.catalog_path, index=False)
        return cfg.catalog_path
    except Exception:  # pragma: no cover - depends on optional pyarrow
        fallback = cfg.catalog_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def read_prepared_catalog(cfg: Config) -> pd.DataFrame:
    """Read back a catalog previously written by :func:`save_catalog`."""
    if cfg.catalog_path.exists():
        return pd.read_parquet(cfg.catalog_path)
    fallback = cfg.catalog_path.with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback, dtype={ID_COLUMN: str})
    raise FileNotFoundError(
        "No prepared catalog found. Run `fashion-retrieval prepare` first."
    )
