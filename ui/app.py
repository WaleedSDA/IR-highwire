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
st.title("BoolFellas — Biomedical Information Retrieval")
st.caption("Highwire Press · TREC Genomics 2006–2007")

search_tab, feedback_tab, eval_tab = st.tabs(["Search", "Relevance Feedback", "Evaluation"])

# ------------------------------------------------------------------
# Search UI
# ------------------------------------------------------------------
with search_tab:
    query = st.text_input(
        "Query",
        placeholder='e.g.  gene expression   |   "DNA repair"   |   gene* cancer   |   #5(gene expression)',
        key="search_query",
    )

    col1, col2, col3, col4 = st.columns(4)
    use_mesh = col1.checkbox("MeSH expansion", help="Expand terms via NCBI MeSH vocabulary")
    use_feedback = col2.checkbox("Relevance feedback (Bo1/KL)")
    use_neural = col3.checkbox("Neural re-rank (BioBERT)", value=True)
    ranker = col4.selectbox("1st-stage ranker", ["bm25", "tfidf"])

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
                        "use_neural": use_neural,
                        "ranker": ranker,
                        "top_k": top_k,
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
        for i, hit in enumerate(hits, 1):
            with st.expander(f"{i}. `{hit['docno']}` — score: {hit['score']:.4f}"):
                snippet = hit.get("snippet", "")
                st.markdown(snippet if snippet else "_No snippet available._")

# ------------------------------------------------------------------
# Feedback UI
# ------------------------------------------------------------------
with feedback_tab:
    st.markdown("Submit a query to apply pseudo-relevance feedback (Bo1/KL) and see the refined results.")
    fb_query = st.text_input("Query", key="fb_query")

    if st.button("Apply Feedback", type="primary") and fb_query.strip():
        with st.spinner("Applying relevance feedback…"):
            try:
                resp = requests.post(
                    f"{API_URL}/feedback",
                    json={"query": fb_query, "relevant_doc_ids": []},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")
                st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        st.subheader("Feedback-refined results")
        for i, hit in enumerate(data.get("results", []), 1):
            st.write(f"{i}. `{hit['docno']}` — score: {hit['score']:.4f}")

# ------------------------------------------------------------------
# Eval UI
# ------------------------------------------------------------------
with eval_tab:
    st.markdown(
        "Runs `pt.Experiment` over BM25 and TF-IDF against the official Highwire TREC qrels. "
        "This may take several minutes the first time."
    )

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running pt.Experiment — this takes a few minutes…"):
            try:
                resp = requests.get(f"{API_URL}/evaluate", timeout=600)
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
