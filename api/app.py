from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.search_engine import SearchEngine

app = FastAPI(title="BoolFellas IR System", version="1.0")

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
    use_neural: bool = True
    ranker: str = "bm25"
    top_k: int = 10


class FeedbackRequest(BaseModel):
    query: str
    relevant_doc_ids: list[str] = []


class SearchResult(BaseModel):
    docno: str
    score: float
    snippet: str = ""


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest) -> dict:
    try:
        results = _engine.search(
            raw_query=req.query,
            use_mesh=req.use_mesh,
            use_feedback=req.use_feedback,
            use_neural=req.use_neural,
            ranker=req.ranker,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top = results.head(req.top_k)
    records = []
    for _, row in top.iterrows():
        records.append({
            "docno": str(row["docno"]),
            "score": float(row["score"]),
            "snippet": str(row.get("snippet", "")),
        })
    return {"query": req.query, "results": records}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Re-retrieve with pseudo-relevance feedback applied."""
    try:
        results = _engine.search(
            raw_query=req.query,
            use_feedback=True,
            use_neural=False,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top = results.head(10)
    records = [
        {"docno": str(row["docno"]), "score": float(row["score"])}
        for _, row in top.iterrows()
    ]
    return {"query": req.query, "results": records}


@app.get("/evaluate")
def evaluate() -> list[dict]:
    """Run pt.Experiment and return metrics for BM25 and TF-IDF."""
    try:
        df = _engine.evaluate()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return df.to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": _engine._initialized}
