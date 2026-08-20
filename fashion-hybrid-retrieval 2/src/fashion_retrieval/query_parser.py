"""Query-understanding stage (Section 4.4).

Decomposes a free-form shopper query into

* **hard filters** -- categorical constraints expressed as field/value pairs
  drawn from the catalog's closed vocabulary, and
* **soft intent** -- the residual stylistic language used only for ranking.

Two design decisions follow from the failure mode identified in Section 3.4,
in which a wrong filter removes the correct product before ranking can occur:

1. the prompt instructs the model to emit a filter only for an explicitly
   stated constraint, so that ambiguity defaults to inclusion;
2. every emitted value is validated against the vocabulary, and unknown
   values are dropped rather than passed to the filter, since a value the
   catalog does not use can only ever match nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .llm import LLMClient, extract_json

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a query-understanding component inside a \
fashion e-commerce search engine.

Your task is to decompose a shopper's search query into two parts:

1. HARD FILTERS: categorical constraints that the shopper stated EXPLICITLY \
and that any acceptable product must satisfy. Each filter must use one of the \
fields below, and its value must be copied EXACTLY from that field's list of \
allowed values.
2. SOFT INTENT: everything else in the query -- style, mood, occasion, fit, \
material impressions -- as a short natural-language phrase used for ranking.

Rules:
- Emit a filter ONLY when the constraint is explicit. If you are unsure, or if \
the query merely implies something, leave the field out. A missing filter is \
much cheaper than a wrong one, because a wrong filter removes correct products \
from consideration entirely.
- Never invent a value. If the shopper asks for something that is not in the \
allowed values, leave that field out and keep the wording in the soft intent.
- Report a confidence between 0 and 1 for each filter you emit.
- Respond with a single JSON object and nothing else, in this exact shape:
{{"hard_filters": {{"field": "value"}}, "soft_intent": "...", \
"confidence": {{"field": 0.0}}}}

The allowed fields and values are:
<vocabulary>
{vocabulary}
</vocabulary>
"""


@dataclass
class ParsedQuery:
    """Structured representation of a shopper query."""

    raw_query: str
    hard_filters: dict[str, str] = field(default_factory=dict)
    soft_intent: str = ""
    confidence: dict[str, float] = field(default_factory=dict)
    dropped_values: dict[str, str] = field(default_factory=dict)
    parse_failed: bool = False

    @property
    def ranking_text(self) -> str:
        """Text handed to the ranking stage.

        Falls back to the raw query when the soft intent is empty, which
        happens when the query consists entirely of hard constraints; ranking
        on an empty string would otherwise be meaningless.
        """
        return self.soft_intent.strip() or self.raw_query

    def ordered_filters(self) -> list[tuple[str, str]]:
        """Filters sorted by descending confidence (relaxation order)."""
        return sorted(
            self.hard_filters.items(),
            key=lambda kv: self.confidence.get(kv[0], 0.5),
            reverse=True,
        )


class QueryParser:
    """Wraps an LLM into the query-understanding component."""

    def __init__(self, llm: LLMClient, vocabulary: dict[str, list[str]]):
        self.llm = llm
        self.vocabulary = vocabulary
        self._lookup = {
            fld: {str(v).lower(): str(v) for v in values}
            for fld, values in vocabulary.items()
        }
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            vocabulary=json.dumps(vocabulary, ensure_ascii=False, indent=2)
        )

    def parse(self, query: str) -> ParsedQuery:
        raw = self.llm.complete(self.system_prompt, query)
        payload = extract_json(raw)
        if not payload:
            log.warning("Query parser returned no usable JSON for: %s", query)
            return ParsedQuery(raw_query=query, soft_intent=query, parse_failed=True)
        return self._validate(query, payload)

    def _validate(self, query: str, payload: dict) -> ParsedQuery:
        """Keep only field/value pairs the catalog can actually match."""
        hard: dict[str, str] = {}
        dropped: dict[str, str] = {}
        raw_filters = payload.get("hard_filters") or {}
        if not isinstance(raw_filters, dict):
            raw_filters = {}

        for fld, value in raw_filters.items():
            if fld not in self._lookup:
                dropped[str(fld)] = str(value)
                continue
            canonical = self._lookup[fld].get(str(value).strip().lower())
            if canonical is None:
                dropped[str(fld)] = str(value)
                continue
            hard[fld] = canonical

        raw_conf = payload.get("confidence") or {}
        confidence = {
            f: float(raw_conf.get(f, 0.5))
            for f in hard
            if isinstance(raw_conf.get(f, 0.5), (int, float))
        }

        soft = payload.get("soft_intent") or ""
        return ParsedQuery(
            raw_query=query,
            hard_filters=hard,
            soft_intent=str(soft),
            confidence=confidence,
            dropped_values=dropped,
        )
