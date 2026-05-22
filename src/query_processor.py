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
        """Expand every term in the query through MeSH synonyms and related terms."""
        expanded: list[str] = []
        for term in q.terms:
            expanded.extend(self.mesh_expander.expand(term))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in expanded:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                unique.append(t)
        q.processed = " ".join(unique)
        q.terms = unique
        return q

    def apply_feedback(self, q: Query, docs: pd.DataFrame) -> Query:
        """Expand query using pseudo-relevance feedback from top-ranked docs."""
        if self._relevance_feedback is None or docs is None or docs.empty:
            return q
        q.processed = self._relevance_feedback.apply_feedback(q.processed, docs)
        q.terms = [
            t for t in re.split(r"\s+", re.sub(r"[^\w\s]", " ", q.processed)) if t
        ]
        return q
