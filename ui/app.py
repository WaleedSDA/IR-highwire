"""
BoolFellas — Streamlit Search UI
Tabs: Search UI | Feedback UI | Eval UI
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="BoolFellas IR", page_icon="🔬", layout="wide")
st.title("BoolFellas - Implementation of an Information Retrieval System for Medical Scientific Documents")
st.caption("Highwire Press · TREC Genomics 2006–2007")

search_tab, eval_tab = st.tabs(["Search", "Evaluation"])

# ------------------------------------------------------------------
# Search UI
# ------------------------------------------------------------------
with search_tab:
    query = st.text_input(
        "Query",
        placeholder='e.g.  cancer AND therapy   |   cancer NOT chemotherapy   |   "DNA repair"   |   gene* cancer   |   #5(gene expression)',
        key="search_query",
    )

    col1, col2, col3, col4 = st.columns(4)
    use_mesh = col1.checkbox("MeSH expansion (Offline)", help="Expand terms instantly using the local compiled SQLite MeSH database")
    use_feedback = col2.checkbox("Relevance feedback")
    feedback_model = col2.selectbox("Feedback model", ["Bo1", "KL"], disabled=not use_feedback, label_visibility="collapsed", help="Bo1: Bose-Einstein 1 · KL: Kullback-Leibler divergence")
    use_neural = col3.checkbox("Neural re-rank", value=False)
    neural_model = col3.selectbox("Neural Model", ["biobert", "pubmedbert"], disabled=not use_neural, label_visibility="collapsed")
    
    neural_top_k = 100
    if use_neural:
        neural_top_k = col3.number_input("Rerank top K", min_value=10, max_value=500, value=100, step=10, help="Number of top documents from BM25/TF-IDF to rerank using Neural model")

    ranker = col4.selectbox("1st-stage ranker", ["bm25", "tfidf"])

    bm25_k1 = None
    bm25_b = None
    if ranker == "bm25":
        sub_col1, sub_col2 = col4.columns(2)
        bm25_k1 = sub_col1.number_input("k1", min_value=0.0, max_value=10.0, value=1.5, step=0.1, help="BM25 term frequency saturation parameter")
        bm25_b = sub_col2.number_input("b (beta)", min_value=0.0, max_value=1.0, value=0.75, step=0.05, help="BM25 document length normalization parameter")

    top_k = st.slider("Results to show", min_value=5, max_value=50, value=10, step=5)

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Retrieving…"):
            try:
                resp = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query": query,
                        "use_mesh": use_mesh,
                        "use_feedback": use_feedback,
                        "feedback_model": feedback_model,
                        "use_neural": use_neural,
                        "neural_model": neural_model,
                        "ranker": ranker,
                        "top_k": top_k,
                        "bm25_k1": bm25_k1,
                        "bm25_b": bm25_b,
                        "neural_top_k": neural_top_k,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server. Is it running?")
                st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        hits = data.get("results", [])
        st.subheader(f"{len(hits)} results for: _{data['query']}_")

        expanded_query = data.get("expanded_query")
        if expanded_query and expanded_query != data["query"]:
            st.info(f"**Expanded query:** {expanded_query}")

        for i, hit in enumerate(hits, 1):
            title_text = hit.get("title", "").strip() or "No Title Available"
            journal_text = hit.get("journal", "").strip().upper() or "UNKNOWN"
            
            title_truncated = f"{title_text[:120]}..." if len(title_text) > 120 else title_text
            with st.expander(f"{i}. `{hit['docno']}` — Score: {hit['score']:.4f} — {title_truncated}"):
                st.markdown(f"#### **Title:** {title_text}")
                st.markdown(f"**Journal:** `{journal_text}`")
                st.divider()
                
                snippet = hit.get("snippet", "")
                st.markdown("**Snippet Preview:**")
                st.markdown(snippet if snippet else "_No snippet available._")
                
                full_text = hit.get("text", "")
                if full_text:
                    st.write("")
                    with st.expander("📄 View Full Document Text"):
                        st.markdown(full_text)

# ------------------------------------------------------------------
# Eval UI
# ------------------------------------------------------------------
with eval_tab:
    st.markdown(
        "Runs `pt.Experiment` over BM25 and TF-IDF against the official Highwire TREC qrels. "
        "This may take several minutes the first time."
    )

    col_e1, col_e2 = st.columns(2)
    use_neural_eval = col_e1.checkbox(
        "Include Neural re-rank (BioBERT)",
        value=False,
        help="Adds +Neural and +Bo1+Neural rows. Runs BioBERT over all 56 topics — expect ~20 extra minutes.",
    )
    use_mesh_eval = col_e2.checkbox(
        "Include MeSH expansion (Offline)",
        value=False,
        help="Include MeSH-enabled query expansion pipelines in the evaluation.",
    )

    bm25_variants = ["BM25(k1=1.2,b=0.75)", "BM25(k1=1.5,b=0.75)", "BM25(k1=2.0,b=0.75)", "BM25(k1=1.5,b=0.30)", "BM25(k1=1.5,b=1.00)"]
    base_rankers = bm25_variants + ["TF-IDF"]
    pipelines = base_rankers[:]
    pipelines += [f"{r}+{fb}" for r in base_rankers for fb in ("Bo1", "KL")]
    if use_neural_eval:
        pipelines += [f"{r}+Neural" for r in base_rankers]
        pipelines += [f"{r}+{fb}+Neural" for r in base_rankers for fb in ("Bo1", "KL")]
    if use_mesh_eval:
        pipelines += [f"{p}+MeSH" for p in pipelines]
    st.caption(f"Will evaluate {len(pipelines)} pipelines: {', '.join(pipelines[:6])} …")

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running pt.Experiment — this may take several minutes…"):
            try:
                resp = requests.get(
                    f"{API_URL}/evaluate",
                    params={
                        "use_neural": str(use_neural_eval).lower(),
                        "use_mesh": str(use_mesh_eval).lower()
                    },
                    timeout=3600,
                )
                resp.raise_for_status()
                records = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")
                st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        import pandas as pd
        df = pd.DataFrame(records)
        st.subheader("Evaluation results")
        st.dataframe(df, use_container_width=True)
