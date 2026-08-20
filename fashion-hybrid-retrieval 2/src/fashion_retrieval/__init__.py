"""Hybrid retrieval for fashion e-commerce search.

Companion code for the undergraduate thesis "Hybrid Retrieval for Fashion
E-Commerce Search: Combining Semantic Similarity and LLM-Based Query
Understanding" (INF/UFRGS).
"""

__version__ = "0.1.0"

from .config import Config
from .query_parser import ParsedQuery, QueryParser
from .retrieval import BaselineRetriever, HybridRetriever, SearchResult

__all__ = [
    "Config",
    "ParsedQuery",
    "QueryParser",
    "BaselineRetriever",
    "HybridRetriever",
    "SearchResult",
]
