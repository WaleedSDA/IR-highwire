from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_MAP: dict[str, str] = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}


class NeuralReranker:
    """
    Re-ranks BM25 top-100 candidates using BioBERT or PubMedBERT embeddings.
    Uses cosine similarity between query and document embeddings.
    """

    def __init__(self, model_name: str = "biobert"):
        from sentence_transformers import SentenceTransformer

        hf_id = MODEL_MAP.get(model_name.lower(), model_name)
        self.model = SentenceTransformer(hf_id)
        self._model_name = model_name
        self._embedding_cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for idx, text in enumerate(texts):
            if text in self._embedding_cache:
                results[idx] = self._embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if uncached_texts:
            encoded = self.model.encode(
                uncached_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for idx, text, emb in zip(uncached_indices, uncached_texts, encoded):
                self._embedding_cache[text] = emb
                results[idx] = emb

        return np.array(results)

    def semantic_score(self, doc_text: str, query: str) -> float:
        embs = self.embed([query, doc_text])
        # embeddings are already L2-normalised so dot product == cosine similarity
        return float(np.dot(embs[0], embs[1]))

    def rerank(self, candidates: pd.DataFrame, query: str) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        texts = candidates["text"].fillna("").tolist()
        doc_embs = self.embed(texts)
        q_emb = self.embed([query])[0]

        # Cosine similarity (already normalised → plain dot product)
        scores = doc_embs @ q_emb

        result = candidates.copy()
        result["score"] = scores
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    def as_transformer(self):
        """Wrap this reranker as a PyTerrier Transformer for use in pt.Experiment."""
        import pyterrier as pt

        _self = self

        class _NeuralTransformer(pt.Transformer):
            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                if df.empty:
                    return df
                if "text" not in df.columns:
                    df = df.copy()
                    df["text"] = ""
                results = []
                for _, group in df.groupby("qid", sort=False):
                    query = group["query"].iloc[0]
                    results.append(_self.rerank(group.copy(), query))
                return pd.concat(results).reset_index(drop=True) if results else df

        return _NeuralTransformer()
