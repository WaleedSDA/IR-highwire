from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .mesh_expander import MeSHExpander
from .relevance_feedback import RelevanceFeedback
from .wildcard_handler import WildcardHandler


@dataclass
class Query:
    raw: str
    processed: str = ""
    expanded_query: str = ""  # populated after MeSH or pseudo-RF expansion
    terms: list[str] = field(default_factory=list)
    is_phrase: bool = False
    is_proximity: bool = False
    proximity_window: int = 0

    def __post_init__(self):
        if not self.processed:
            self.processed = self.raw


class QueryProcessor:
    """
    Parses, expands (MeSH), and applies relevance feedback to queries.
    Attributes: meshExpander: MeSHExpander, feedbackModel: RelevanceFeedback.
    """

    def __init__(
        self,
        index_ref=None,
        terrier_index=None,
        mesh_email: str = "user@example.com",
        feedback_model: str = "Bo1",
    ):
        self.mesh_expander = MeSHExpander(email=mesh_email)
        self._feedback_model_name = feedback_model
        self._index_ref = index_ref
        self._terrier_index = terrier_index

        self._relevance_feedback: RelevanceFeedback | None = None
        self._wildcard_handler: WildcardHandler | None = None

        if index_ref is not None and terrier_index is not None:
            self._init_components(index_ref, terrier_index)

    def _init_components(self, index_ref, terrier_index) -> None:
        self._relevance_feedback = RelevanceFeedback(index_ref, self._feedback_model_name)
        self._wildcard_handler = WildcardHandler(terrier_index)

    def set_index(self, index_ref, terrier_index) -> None:
        self._index_ref = index_ref
        self._terrier_index = terrier_index
        self._init_components(index_ref, terrier_index)

    @property
    def feedback_model(self) -> RelevanceFeedback | None:
        return self._relevance_feedback

    def parse_query(self, raw: str) -> Query:
        q = Query(raw=raw)

        # Phrase query: "term1 term2"
        if raw.startswith('"') and raw.endswith('"') and len(raw) > 2:
            q.is_phrase = True
            q.processed = raw  # Terrier handles phrase natively

        # Proximity query: #N(term1 term2)
        prox = re.match(r"^#(\d+)\((.+)\)$", raw.strip())
        if prox:
            q.is_proximity = True
            q.proximity_window = int(prox.group(1))
            q.processed = raw  # Terrier handles proximity natively

        # Wildcard expansion
        if self._wildcard_handler and ("*" in raw or "?" in raw) and not q.is_phrase:
            q.processed = self._wildcard_handler.expand_query(raw)

        q.terms = [
            t for t in re.split(r"\s+", re.sub(r'[^\w\s]', " ", q.processed)) if t
        ]
        return q

    def expand_with_mesh(self, q: Query) -> Query:
        """Expand via MeSH: try the full query as one concept first, then per-term."""
        # Try the whole query as a single MeSH descriptor (e.g. "DNA repair" → D004260).
        # _fetch_mesh_data returns only the original term when no descriptor is found,
        # so len > 1 means a real match was found.
        full_expansion = self.mesh_expander.expand(q.processed)
        if len(full_expansion) > 1:
            raw_entries = full_expansion
        else:
            # Fall back: expand each token independently.
            raw_entries = []
            for term in q.terms:
                raw_entries.extend(self.mesh_expander.expand(term))

        # MeSH entries contain commas, hyphens, parentheses, etc. that break
        # Terrier's query parser — tokenise each entry into plain words.
        tokens: list[str] = []
        seen: set[str] = set()
        for entry in raw_entries:
            for tok in re.split(r"\s+", re.sub(r"[^\w\s]", " ", entry)):
                tl = tok.lower()
                if tok and tl not in seen:
                    seen.add(tl)
                    tokens.append(tok)

        q.processed = " ".join(tokens)
        q.expanded_query = q.processed
        q.terms = tokens
        return q

    def apply_feedback(self, q: Query, docs: pd.DataFrame) -> Query:
        """Expand query using pseudo-relevance feedback from top-ranked docs."""
        if self._relevance_feedback is None or docs is None or docs.empty:
            return q
        q.processed = self._relevance_feedback.apply_feedback(q.processed, docs)
        q.expanded_query = q.processed
        q.terms = [
            t for t in re.split(r"\s+", re.sub(r"[^\w\s]", " ", q.processed)) if t
        ]
        return q
