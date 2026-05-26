from __future__ import annotations

import pandas as pd


class RelevanceFeedback:
    """
    Wraps PyTerrier's Bo1 and KL query expansion models.
    Takes top-ranked feedback documents and expands the query.
    """

    def __init__(self, index_ref, model: str = "Bo1"):
        from .pt_initializer import init_pyterrier
        init_pyterrier()

        import pyterrier as pt


        self.model_name = model
        self._index_ref = index_ref
        if model == "Bo1":
            self._rewriter = pt.rewrite.Bo1QueryExpansion(index_ref)
        elif model == "KL":
            self._rewriter = pt.rewrite.KLQueryExpansion(index_ref)
        else:
            raise ValueError(f"Unknown feedback model: {model!r}. Choose 'Bo1' or 'KL'.")

    def apply_feedback(self, query: str, docs: pd.DataFrame) -> str:
        """Expand *query* using the top feedback documents; returns expanded query string."""
        return self.expand_query(query, docs)

    def expand_query(self, query: str, docs: pd.DataFrame) -> str:
        if docs.empty:
            return query

        qid = "0"
        topics = pd.DataFrame([{"qid": qid, "query": query}])

        fb_docs = docs.copy()
        fb_docs["qid"] = qid
        if "rank" not in fb_docs.columns:
            fb_docs["rank"] = range(1, len(fb_docs) + 1)

        # Bo1/KL transform: input must have both topic cols and result cols in one frame
        combined = topics.merge(
            fb_docs[["qid", "docno", "score", "rank"]],
            on="qid",
        )
        try:
            expanded = self._rewriter.transform(combined)
            if expanded.empty:
                return query
            return str(expanded.iloc[0]["query"])
        except Exception:
            return query

    def get_pipeline(self, retriever):
        """Return a full pseudo-relevance feedback pipeline: BM25 → expand → BM25."""
        return retriever >> self._rewriter >> retriever
