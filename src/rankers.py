from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Ranker(ABC):
    """Abstract Ranker interface from the UML."""

    @abstractmethod
    def rank(self, query: str, top_k: int = 100) -> pd.DataFrame:
        pass

    @abstractmethod
    def score(self, doc: dict, query: str) -> float:
        pass

    @abstractmethod
    def get_pipeline(self):
        pass


class BM25Ranker(Ranker):
    """
    BM25 first-stage ranker.
    Attributes: k1=1.5, b=0.75, topK=100 (UML defaults).
    """

    def __init__(self, index_ref, k1: float = 1.5, b: float = 0.75, top_k: int = 100):
        import pyterrier as pt
        if not pt.started():
            pt.init()

        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self._retriever = pt.terrier.Retriever(
            index_ref,
            wmodel="BM25",
            controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
            num_results=top_k,
            metadata=["docno", "text"],
        )

    def rank(self, query: str, top_k: int = None) -> pd.DataFrame:
        self._retriever.num_results = top_k or self.top_k
        return self._retriever.search(query)

    def score(self, doc: dict, query: str) -> float:
        raise NotImplementedError("Use rank() for batch scoring via PyTerrier.")

    def get_pipeline(self):
        return self._retriever


class TFIDFRanker(Ranker):
    """
    TF-IDF alternative first-stage ranker.
    Attribute: normType (passed as wmodel variant).
    """

    def __init__(self, index_ref, norm_type: str = "TF_IDF", top_k: int = 100):
        import pyterrier as pt
        if not pt.started():
            pt.init()

        self.norm_type = norm_type
        self.top_k = top_k
        self._retriever = pt.terrier.Retriever(
            index_ref,
            wmodel="TF_IDF",
            num_results=top_k,
            metadata=["docno", "text"],
        )

    def rank(self, query: str, top_k: int = None) -> pd.DataFrame:
        self._retriever.num_results = top_k or self.top_k
        return self._retriever.search(query)

    def score(self, doc: dict, query: str) -> float:
        raise NotImplementedError("Use rank() for batch scoring via PyTerrier.")

    def get_pipeline(self):
        return self._retriever
