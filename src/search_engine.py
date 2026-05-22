from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from .index import PositionalIndex
from .neural_reranker import NeuralReranker
from .query_processor import QueryProcessor
from .rankers import BM25Ranker, Ranker, TFIDFRanker
from .snippet_generator import SnippetGenerator
from .evaluation import EvaluationEngine


@dataclass
class SearchResponse:
    results: pd.DataFrame
    expanded_query: str = ""  # non-empty only when MeSH or pseudo-RF rewrote the query


class SearchEngine:
    """
    Orchestrator — wires together all IR components.
    Pipeline: query → parse/expand → BM25(100) → NeuralReranker → snippets.
    Attributes (from UML): queryProc, index, rankers, reranker.
    """

    def __init__(
        self,
        index_path: str,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        top_k: int = 100,
        neural_model: str = "biobert",
        mesh_email: str = "user@example.com",
        feedback_model: str = "Bo1",
    ):
        self.index = PositionalIndex(index_path=index_path)
        self._cfg = dict(
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            top_k=top_k,
            neural_model=neural_model,
            mesh_email=mesh_email,
            feedback_model=feedback_model,
        )

        # Populated after index is ready
        self.query_proc: QueryProcessor | None = None
        self.rankers: list[Ranker] = []
        self.reranker: NeuralReranker | None = None
        self._snippet_gen: SnippetGenerator | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(self, docs: Iterator[dict]) -> None:
        """Index the corpus from scratch and initialise all components."""
        self.index.index(docs)
        self._post_index_init()

    def load_index(self) -> None:
        """Load an existing on-disk index and initialise all components."""
        self.index.load()
        self._post_index_init()

    def _post_index_init(self) -> None:
        cfg = self._cfg
        index_ref = self.index.index_ref
        terrier_index = self.index.terrier_index

        self.query_proc = QueryProcessor(
            index_ref=index_ref,
            terrier_index=terrier_index,
            mesh_email=cfg["mesh_email"],
            feedback_model=cfg["feedback_model"],
        )
        self.rankers = [
            BM25Ranker(index_ref, k1=cfg["bm25_k1"], b=cfg["bm25_b"], top_k=cfg["top_k"]),
            TFIDFRanker(index_ref, top_k=cfg["top_k"]),
        ]
        self.reranker = NeuralReranker(model_name=cfg["neural_model"])
        self._snippet_gen = SnippetGenerator(index_ref)
        self._initialized = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        raw_query: str,
        use_mesh: bool = False,
        use_feedback: bool = False,
        use_neural: bool = True,
        ranker: str = "bm25",
    ) -> SearchResponse:
        """
        Full retrieval pipeline.
        1. Parse query (phrase / proximity / wildcard detection).
        2. Optionally expand with MeSH.
        3. First-stage retrieval: BM25 or TF-IDF top-100.
        4. Optionally apply pseudo-relevance feedback (Bo1/KL — no user labels needed).
        5. Optionally neural re-rank with BioBERT/PubMedBERT.
        6. Attach snippets; preserve full document text in results.
        """
        self._require_init()

        query = self.query_proc.parse_query(raw_query)

        if use_mesh:
            query = self.query_proc.expand_with_mesh(query)

        active_ranker = self.rankers[0] if ranker.lower() == "bm25" else self.rankers[1]
        candidates = active_ranker.rank(query.processed, top_k=self._cfg["top_k"])

        if use_feedback and not candidates.empty:
            query = self.query_proc.apply_feedback(query, candidates)
            candidates = active_ranker.rank(query.processed, top_k=self._cfg["top_k"])

        if "text" not in candidates.columns:
            candidates["text"] = ""

        if use_neural and not candidates.empty:
            candidates = self.reranker.rerank(candidates, query.processed)

        if not candidates.empty:
            candidates["snippet"] = candidates.apply(
                lambda row: self._snippet_gen.generate(
                    str(row.get("text", "")), query.processed
                ),
                axis=1,
            )

        return SearchResponse(results=candidates, expanded_query=query.expanded_query)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_names: list[str] | None = None,
        use_feedback: bool = True,
        use_neural: bool = False,
    ) -> pd.DataFrame:
        """
        Run pt.Experiment over all enabled pipeline combinations.

        use_neural defaults to False because BioBERT reranking over all
        56 TREC topics takes ~20 minutes; enable it deliberately.
        """
        self._require_init()
        engine = EvaluationEngine(dataset_names)
        return engine.run_experiment(
            rankers=self.rankers,
            names=["BM25", "TF-IDF"],
            feedback=self.query_proc.feedback_model if use_feedback else None,
            reranker=self.reranker if use_neural else None,
        )

    def _require_init(self) -> None:
        if not self._initialized:
            raise RuntimeError("Index not ready. Call build_index() or load_index() first.")
