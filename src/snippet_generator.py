from __future__ import annotations

import re


_SNIPPET_WORDS = 80
_CONTEXT_WINDOW = 30


class SnippetGenerator:
    """
    Generates query-term-highlighted snippets from document text.
    Uses pt.text.snippets under the hood when a pipeline is needed;
    also provides a standalone highlight() method.
    Attribute: indexRef (stored for pt.text.snippets pipeline construction).
    """

    def __init__(self, index_ref):
        self.index_ref = index_ref

    def generate(self, doc_text: str, query: str) -> str:
        """Return the best snippet from doc_text centred around query terms."""
        if not doc_text:
            return ""
        terms = {re.sub(r"[^\w]", "", t.lower()) for t in query.split() if t}
        words = doc_text.split()

        # Find the window with the most query-term hits
        best_start, best_hits = 0, -1
        for i in range(len(words)):
            window = words[i : i + _CONTEXT_WINDOW]
            hits = sum(1 for w in window if re.sub(r"[^\w]", "", w.lower()) in terms)
            if hits > best_hits:
                best_hits, best_start = hits, i

        snippet_words = words[best_start : best_start + _SNIPPET_WORDS]
        return self.highlight(" ".join(snippet_words), query)

    def highlight(self, text: str, query: str) -> str:
        """Wrap query terms in ** for markdown bold."""
        terms = {re.sub(r"[^\w]", "", t.lower()) for t in query.split() if t}
        output: list[str] = []
        for word in text.split():
            clean = re.sub(r"[^\w]", "", word.lower())
            output.append(f"**{word}**" if clean in terms else word)
        return " ".join(output)

    def get_pipeline(self):
        """Return a pt.text.snippets transformer for use in PyTerrier pipelines."""
        from .pt_initializer import init_pyterrier
        init_pyterrier()
        import pyterrier as pt

        return pt.text.snippets(self.index_ref)
