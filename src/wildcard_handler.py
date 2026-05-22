from __future__ import annotations

import re


class WildcardHandler:
    """
    Expands wildcard patterns against the index lexicon.
    Uses lexicon expansion rather than a permuterm index to avoid storage overhead.
    """

    def __init__(self, terrier_index):
        self._lexicon = terrier_index.getLexicon()

    def expand(self, pattern: str) -> list[str]:
        """Expand a wildcard pattern (*, ?) against the lexicon."""
        regex_pattern = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
        compiled = re.compile(regex_pattern, re.IGNORECASE)
        matches = []
        for entry in self._lexicon:
            term = entry.getKey()
            if compiled.match(term):
                matches.append(term)
        return matches

    def get_matches(self, query: str) -> list[str]:
        """Return all tokens with wildcards expanded."""
        tokens = query.split()
        expanded: list[str] = []
        for token in tokens:
            if "*" in token or "?" in token:
                expanded.extend(self.expand(token))
            else:
                expanded.append(token)
        return expanded

    def expand_query(self, query: str) -> str:
        """Return the query string with wildcards replaced by matched terms."""
        matches = self.get_matches(query)
        return " ".join(matches) if matches else query
