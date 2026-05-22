# BoolFellas — Biomedical Information Retrieval System

**CENG596 Project** · Furkan Safa Altunyuva & Waleed Shanaa · METU

A multi-stage IR pipeline over the [Highwire Press](https://ir-datasets.com/highwire.html) corpus (162,259 full-text biomedical articles, TREC Genomics 2006–2007). Combines classical retrieval (BM25, TF-IDF) with neural re-ranking (BioBERT / PubMedBERT) and a Streamlit search UI.

---

## Architecture

```
Query
  │
  ▼
QueryProcessor          ← phrase / proximity / wildcard detection
  │   ├─ MeSHExpander   ← NCBI E-utilities synonym/related expansion
  │   └─ RelevanceFeedback (Bo1 / KL)
  ▼
PositionalIndex         ← blocks=True (phrase & proximity), field index (title/journal/text)
  │
  ▼
BM25Ranker / TFIDFRanker  ← top-100 candidates
  │
  ▼
NeuralReranker          ← BioBERT or PubMedBERT cosine re-ranking
  │
  ▼
SnippetGenerator        ← best-window extraction + query-term highlighting
  │
  ▼
Results
```

Component → source file mapping:

| Component | File |
|---|---|
| `PositionalIndex` | `src/index.py` |
| `WildcardHandler` | `src/wildcard_handler.py` |
| `BM25Ranker`, `TFIDFRanker` | `src/rankers.py` |
| `NeuralReranker` | `src/neural_reranker.py` |
| `MeSHExpander` | `src/mesh_expander.py` |
| `RelevanceFeedback` | `src/relevance_feedback.py` |
| `SnippetGenerator` | `src/snippet_generator.py` |
| `QueryProcessor` | `src/query_processor.py` |
| `EvaluationEngine` | `src/evaluation.py` |
| `SearchEngine` (orchestrator) | `src/search_engine.py` |
| REST API | `api/app.py` |
| Streamlit UI | `ui/app.py` |

---

## Quick Start (Docker)

```bash
# Build and start API + UI
docker compose up --build

# First run: build the index (downloads ~162 k articles, takes ~30 min)
docker compose run api python main.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

Two named volumes are created automatically:
- `index` — persists the on-disk Terrier index across restarts
- `pyterrier_cache` — caches downloaded Terrier JARs across rebuilds

---

## Local Setup (without Docker)

Requires **Python 3.13** and **Java 17+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build index
python main.py

# Start API
uvicorn api.app:app --port 8000

# Start UI (separate terminal)
streamlit run ui/app.py
```

---

## Query Syntax

| Type | Example |
|---|---|
| Keyword | `gene expression cancer` |
| Phrase | `"DNA repair mechanism"` |
| Proximity (within N words) | `#5(gene cancer)` |
| Wildcard | `gene* expression` |

Options available via UI checkboxes or API flags:
- **MeSH expansion** — expands each query term with NCBI MeSH synonyms and related terms
- **Relevance feedback** — applies Bo1 or KL pseudo-relevance feedback after first-stage retrieval
- **Neural re-ranking** — re-scores top-100 BM25 candidates with BioBERT or PubMedBERT

---

## REST API

### `POST /search`
```json
{
  "query": "gene expression regulation",
  "use_mesh": false,
  "use_feedback": false,
  "use_neural": true,
  "ranker": "bm25",
  "top_k": 10
}
```

### `POST /feedback`
```json
{
  "query": "gene expression regulation",
  "relevant_doc_ids": []
}
```

### `GET /evaluate`
Runs `pt.Experiment` over BM25 and TF-IDF against the official TREC qrels. Returns MAP, NDCG@10, P@5, P@10, MRR, R-Prec.

### `GET /health`
```json
{ "status": "ok", "index_ready": true }
```

---

## Evaluation Results (Progress Report)

Highwire corpus · 64 queries (TREC Genomics 2006–2007)

| Model | MAP | R-Prec | MRR | P@5 | P@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| TF-IDF | **0.1516** | **0.1792** | **0.4140** | **0.2594** | **0.2172** | **0.2459** |
| BM25 (k1=1.5, b=0.75) | 0.1039 | 0.1138 | 0.3003 | 0.1500 | 0.1422 | 0.1584 |
| BM25 (k1=1.5, b=0.25) | 0.1048 | 0.1167 | 0.3103 | 0.1625 | 0.1438 | 0.1593 |
| BM25 (k1=3.0, b=0.25) | 0.1002 | 0.1166 | 0.3312 | 0.1781 | 0.1516 | 0.1701 |

TF-IDF outperforms all BM25 configurations. BM25's length normalisation penalises the long full-text articles (avg. 6,542 tokens) in this corpus. Neural re-ranking, MeSH expansion, and relevance feedback results will be reported in the final submission.

---

## Index CLI

```bash
# Build index (first time or after corpus changes)
python main.py

# Force rebuild
python main.py --rebuild

# Build index then run pt.Experiment
python main.py --evaluate
```

---

## Dataset

The [Highwire Press](https://ir-datasets.com/highwire.html) collection is fetched automatically via `ir_datasets` on first run:
- 162,259 full-text articles from 49 biomedical journals
- ~994 million words / ~1.06 billion tokens after PyTerrier tokenisation
- 64 TREC queries with official relevance judgements (28 from 2006, 36 from 2007)

---

## References

1. Highwire Press Medical Document Collection, TREC Genomics Track 2006–2007. https://ir-datasets.com/highwire.html
2. Gu et al. (2021). PubMedBERT. https://arxiv.org/abs/2007.15779
3. Lee et al. (2019). BioBERT. https://arxiv.org/abs/1901.08746
4. Macdonald & Tonellotto (2020). PyTerrier. ACM SIGIR ICTIR.
5. National Library of Medicine. Medical Subject Headings (MeSH). https://www.ncbi.nlm.nih.gov/mesh/
