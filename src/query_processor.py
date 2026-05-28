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
    field_constraints: dict[str, list[str]] = field(default_factory=dict)
    is_boolean: bool = False
    boolean_must: list[str] = field(default_factory=list)    # AND terms — all must appear
    boolean_must_not: list[str] = field(default_factory=list)  # NOT terms — none may appear

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
        self._feedback_cache: dict[str, RelevanceFeedback] | None = None
        self._wildcard_handler: WildcardHandler | None = None

        if index_ref is not None and terrier_index is not None:
            self._init_components(index_ref, terrier_index)

    def _init_components(self, index_ref, terrier_index) -> None:
        self._feedback_cache = {
            "Bo1": RelevanceFeedback(index_ref, "Bo1"),
            "KL": RelevanceFeedback(index_ref, "KL"),
        }
        self._relevance_feedback = self._feedback_cache[self._feedback_model_name]
        self._wildcard_handler = WildcardHandler(terrier_index)

    def set_index(self, index_ref, terrier_index) -> None:
        self._index_ref = index_ref
        self._terrier_index = terrier_index
        self._init_components(index_ref, terrier_index)

    @property
    def feedback_model(self) -> RelevanceFeedback | None:
        return self._relevance_feedback

    # Boolean operator pattern — matches uppercase AND / OR / NOT as whole words
    _BOOL_OP_RE = re.compile(r'\b(AND|OR|NOT)\b')
    # Tokenizer for boolean expressions: quoted phrases, operators, or bare words
    _BOOL_TOKEN_RE = re.compile(r'"[^"]+"|AND\s+NOT|\bAND\b|\bOR\b|\bNOT\b|\S+')

    @staticmethod
    def _parse_boolean_expr(query_str: str):
        """
        Parse a boolean query with uppercase AND / OR / NOT operators.
        Quoted phrases are treated as atomic tokens.
        Returns (bm25_query, must_include, must_exclude).
        - must_include: all positive terms when AND is present; all must appear in doc text.
        - must_exclude: NOT terms; none may appear in doc text.
        - bm25_query: positive terms only, passed to the ranker for scoring.
        """
        # Normalise "AND NOT" → "NOT"
        s = re.sub(r'\bAND\s+NOT\b', ' NOT ', query_str)

        # Tokenize: quoted phrases, uppercase operators, bare words
        token_re = re.compile(r'"[^"]+"|(?<!\w)(AND|OR|NOT)(?!\w)|\S+')
        op_re = re.compile(r'^(AND|OR|NOT)$')  # case-sensitive — operators must be uppercase

        positive = []   # list of (original_token, lowercase_term)
        excluded = []   # lowercase terms that must NOT appear
        has_and = False
        pending_op = None

        for tok in token_re.findall(s):
            if op_re.match(tok):
                if tok == 'AND':
                    has_and = True
                pending_op = tok
                continue
            term_clean = tok.strip('"').lower().strip()
            if not term_clean:
                continue
            if pending_op == 'NOT':
                excluded.append(term_clean)
            else:
                positive.append((tok, term_clean))
            pending_op = None

        bm25_query = ' '.join(orig for orig, _ in positive)
        must_include = [t for _, t in positive] if has_and else []
        return bm25_query, must_include, excluded

    def parse_query(self, raw: str) -> Query:
        q = Query(raw=raw)

        # Parse field constraints (e.g. title:cancer, journal:"nature science")
        matches = re.findall(r'(\w+):(?:([^"\s]+)|"([^"]+)")', raw)
        field_constraints = {}
        for f_name, term1, term2 in matches:
            val = term1 or term2
            field_constraints.setdefault(f_name.lower(), []).append(val.lower())
        q.field_constraints = field_constraints

        # Clean the query string from field constraints for first-stage retrieval
        cleaned_query = re.sub(r'\w+:(?:[^"\s]+|"[^"]+")', '', raw).strip()
        if not cleaned_query and field_constraints:
            all_vals = []
            for vals in field_constraints.values():
                all_vals.extend(vals)
            cleaned_query = " ".join(f'"{v}"' if " " in v else v for v in all_vals)

        processed_query = cleaned_query or raw

        # Phrase query: "term1 term2"
        if processed_query.startswith('"') and processed_query.endswith('"') and len(processed_query) > 2:
            q.is_phrase = True
            q.processed = processed_query  # Terrier handles phrase natively

        # Proximity query: #N(term1 term2)
        elif re.match(r"^#(\d+)\((.+)\)$", processed_query.strip()):
            prox = re.match(r"^#(\d+)\((.+)\)$", processed_query.strip())
            q.is_proximity = True
            q.proximity_window = int(prox.group(1))
            q.processed = processed_query  # Terrier handles proximity natively

        # Boolean query: contains uppercase AND / OR / NOT
        elif self._BOOL_OP_RE.search(processed_query):
            bm25_q, must_include, must_exclude = self._parse_boolean_expr(processed_query)
            if bm25_q:  # only treat as boolean if positive terms remain
                q.is_boolean = True
                q.boolean_must = must_include
                q.boolean_must_not = must_exclude
                q.processed = bm25_q
            else:
                q.processed = processed_query

        else:
            # Wildcard expansion
            if self._wildcard_handler and ("*" in processed_query or "?" in processed_query):
                q.processed = self._wildcard_handler.expand_query(processed_query)
            else:
                q.processed = processed_query

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

    def apply_feedback(self, q: Query, docs: pd.DataFrame, model: str | None = None) -> Query:
        """Expand query using pseudo-relevance feedback from top-ranked docs."""
        fb = self._feedback_cache.get(model or self._feedback_model_name) if self._feedback_cache else self._relevance_feedback
        if fb is None or docs is None or docs.empty:
            return q
        q.processed = fb.apply_feedback(q.processed, docs)
        q.expanded_query = q.processed
        q.terms = [
            t for t in re.split(r"\s+", re.sub(r"[^\w\s]", " ", q.processed)) if t
        ]
        return q
