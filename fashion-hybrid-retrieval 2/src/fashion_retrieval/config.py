"""Central configuration for the hybrid retrieval experiments.

Every path and hyper-parameter used by the pipeline lives here so that an
experiment can be reproduced from a single object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root (this file is at <root>/src/fashion_retrieval/config.py)
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Catalog schema (H&M Personalized Fashion Recommendations, articles.csv)
# ---------------------------------------------------------------------------

#: Column holding the free-text product description used for semantic ranking.
DESCRIPTION_COLUMN = "detail_desc"

#: Column holding the short commercial name of the article.
NAME_COLUMN = "prod_name"

#: Unique identifier of an article.
ID_COLUMN = "article_id"

#: Candidate hard-filter fields. Whether each one is actually used is decided
#: empirically by the label-space analysis (see ``label_space.py``).
CANDIDATE_FILTER_FIELDS = [
    "product_type_name",     # e.g. Dress, Skirt, Trousers, Vest top
    "colour_group_name",     # e.g. Black, Light Pink, Dark Blue
    "perceived_colour_master_name",  # coarser colour, e.g. Black, Pink, Blue
    "index_group_name",      # intended audience, e.g. Ladieswear, Menswear
    "graphical_appearance_name",     # e.g. Solid, All over pattern, Stripe
]

#: Fields appended to the product text so that the baseline has access to the
#: same information the hybrid system filters on (fairness control, Sec. 4.3).
TEXT_CONTEXT_FIELDS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "graphical_appearance_name",
    "garment_group_name",
    "index_group_name",
]


@dataclass
class Config:
    """Runtime configuration for a full experimental run."""

    # -- paths ------------------------------------------------------------
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    results_dir: Path = ROOT / "results"
    articles_file: str = "articles.csv"

    # -- catalog preparation ---------------------------------------------
    #: Articles with a description shorter than this are dropped.
    min_description_chars: int = 20
    #: Optional cap on catalog size, useful for quick local runs. None = all.
    max_articles: int | None = None
    random_seed: int = 42

    # -- embeddings -------------------------------------------------------
    #: "hashing" (pure numpy, no download) or "sentence-transformers".
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 512  # only used by the hashing backend
    embedding_batch_size: int = 256

    # -- LLM ---------------------------------------------------------------
    # All backends are free of charge and free of subscription:
    #   "mock"              offline rule-based stand-in (tests only)
    #   "ollama"            local open-weight model via Ollama (recommended)
    #   "transformers"      local open-weight model via Hugging Face
    #   "openai-compatible" any local server speaking the OpenAI chat shape
    llm_backend: str = "mock"
    llm_model: str = "qwen2.5:3b-instruct"
    #: Base URL for the "ollama" and "openai-compatible" backends.
    llm_host: str | None = None
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0

    # -- hybrid retrieval --------------------------------------------------
    #: If filtering leaves fewer than this many candidates, filters are
    #: relaxed one at a time (Sec. 4.4).
    min_candidates: int = 20
    #: Number of results returned by a search.
    top_k: int = 10
    #: Values of k reported in Recall@k.
    recall_k_values: tuple[int, ...] = (1, 5, 10, 50)

    # -- label space -------------------------------------------------------
    #: A field is only usable as a hard filter if no more than this many
    #: distinct values occur, so that the vocabulary fits in a prompt.
    max_filter_cardinality: int = 200
    #: Values occurring in fewer than this many articles are excluded from
    #: the closed vocabulary offered to the LLM.
    min_value_frequency: int = 5

    # -- synthetic queries -------------------------------------------------
    n_synthetic_queries: int = 200
    #: Fraction of generated queries manually inspected for plausibility.
    plausibility_sample_size: int = 30

    filter_fields: list[str] = field(
        default_factory=lambda: [
            "product_type_name",
            "colour_group_name",
            "index_group_name",
        ]
    )

    def __post_init__(self) -> None:
        for path in (self.processed_dir, self.results_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def articles_path(self) -> Path:
        return Path(self.raw_dir) / self.articles_file

    @property
    def catalog_path(self) -> Path:
        return Path(self.processed_dir) / "catalog.parquet"

    @property
    def embeddings_path(self) -> Path:
        return Path(self.processed_dir) / f"embeddings_{self.embedding_backend}.npy"

    @property
    def vocabulary_path(self) -> Path:
        return Path(self.processed_dir) / "filter_vocabulary.json"

    @property
    def synthetic_queries_path(self) -> Path:
        return Path(self.processed_dir) / "synthetic_queries.json"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a config, letting environment variables override defaults."""
        cfg = cls()
        if os.getenv("FR_EMBEDDING_BACKEND"):
            cfg.embedding_backend = os.environ["FR_EMBEDDING_BACKEND"]
        if os.getenv("FR_LLM_BACKEND"):
            cfg.llm_backend = os.environ["FR_LLM_BACKEND"]
        if os.getenv("FR_LLM_MODEL"):
            cfg.llm_model = os.environ["FR_LLM_MODEL"]
        if os.getenv("FR_LLM_HOST"):
            cfg.llm_host = os.environ["FR_LLM_HOST"]
        if os.getenv("FR_MAX_ARTICLES"):
            cfg.max_articles = int(os.environ["FR_MAX_ARTICLES"])
        return cfg
