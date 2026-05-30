from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel

from src.search_engine import SearchEngine

app = FastAPI(title="BoolFellas IR System", version="1.0")
_log = logging.getLogger(__name__)

INDEX_PATH = os.environ.get("INDEX_PATH", "./index")
_engine = SearchEngine(index_path=INDEX_PATH)

try:
    _engine.load_index()
except FileNotFoundError:
    pass  # Index will be built via /index endpoint or main.py


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    use_mesh: bool = False
    use_feedback: bool = False
    feedback_model: str = "Bo1"
    use_neural: bool = False
    neural_model: str = "biobert"
    ranker: str = "bm25"
    top_k: int = 10
    bm25_k1: float | None = None
    bm25_b: float | None = None
    neural_top_k: int = 100


class FeedbackRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    docno: str
    score: float
    text: str = ""
    snippet: str = ""


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest) -> dict:
    try:
        resp = _engine.search(
            raw_query=req.query,
            use_mesh=req.use_mesh,
            use_feedback=req.use_feedback,
            feedback_model=req.feedback_model,
            use_neural=req.use_neural,
            ranker=req.ranker,
            neural_model=req.neural_model,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b,
            neural_top_k=req.neural_top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.exception("search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    top = resp.results.head(req.top_k)
    records = []
    for _, row in top.iterrows():
        records.append({
            "docno": str(row["docno"]),
            "score": float(row["score"]),
            "title": str(row.get("title", "")),
            "journal": str(row.get("journal", "")),
            "text": str(row.get("text", "")),
            "snippet": str(row.get("snippet", "")),
        })
    return {
        "query": req.query,
        "expanded_query": resp.expanded_query or None,
        "results": records,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Re-retrieve using pseudo-relevance feedback (Bo1 on top docs — no user labels needed)."""
    try:
        resp = _engine.search(
            raw_query=req.query,
            use_feedback=True,
            use_neural=False,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top = resp.results.head(10)
    records = [
        {"docno": str(row["docno"]), "score": float(row["score"])}
        for _, row in top.iterrows()
    ]
    return {
        "query": req.query,
        "expanded_query": resp.expanded_query or None,
        "results": records,
    }


@app.get("/evaluate")
def evaluate(use_neural: bool = False, use_mesh: bool = True) -> list[dict]:
    """
    Run pt.Experiment over all pipeline combinations.
    Pass ?use_neural=true to include BioBERT/PubMedBERT reranking.
    Pass ?use_mesh=true/false to include/exclude MeSH query expansion.
    """
    try:
        df = _engine.evaluate(use_neural=use_neural, use_mesh=use_mesh)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.exception("evaluate failed: %s", e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return df.to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": _engine._initialized}
