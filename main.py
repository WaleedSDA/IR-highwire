#!/usr/bin/env python3
"""
BoolFellas — entry point for index construction and quick smoke-test.

Usage:
    python main.py               # build index if missing, then run a test query
    python main.py --rebuild     # force rebuild index
    python main.py --evaluate    # run pt.Experiment after loading index
"""
from __future__ import annotations

import argparse
import os
import sys

INDEX_PATH = os.environ.get("INDEX_PATH", "./index")
DATASET_2006 = "highwire/trec-genomics-2006"
DATASET_2007 = "highwire/trec-genomics-2007"


def iter_corpus():
    import ir_datasets

    # The 2006 and 2007 Highwire collections share the exact same 162,259 documents.
    # Index only once from the 2006 split to avoid duplicate docno values that break
    # Terrier's MetaIndex reverse-lookup (which requires unique keys).
    ds = ir_datasets.load(DATASET_2006)
    for doc in ds.docs_iter():
        yield {
            "docno": doc.doc_id,
            "text": doc.default_text(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force index rebuild")
    parser.add_argument("--evaluate", action="store_true", help="Run pt.Experiment")
    args = parser.parse_args()

    import pyterrier as pt
    if not pt.started():
        pt.init()

    from src.search_engine import SearchEngine

    engine = SearchEngine(
        index_path=INDEX_PATH,
        bm25_k1=1.5,
        bm25_b=0.75,
        top_k=100,
        neural_model="biobert",
    )

    index_exists = os.path.exists(os.path.join(INDEX_PATH, "data.properties"))

    if args.rebuild or not index_exists:
        print("Building index over Highwire corpus (162 k docs) — this takes ~30 min…")
        engine.build_index(iter_corpus())
        print("Index built.")
    else:
        print(f"Loading existing index from {INDEX_PATH} …")
        engine.load_index()
        print("Index loaded.")

    # Smoke test
    print("\n--- Smoke test: BM25 retrieval (no neural) ---")
    results = engine.search("gene expression regulation cancer", use_neural=False)
    print(results[["docno", "score"]].head(5).to_string(index=False))

    print("\n--- Smoke test: phrase query ---")
    results = engine.search('"gene expression"', use_neural=False)
    print(results[["docno", "score"]].head(5).to_string(index=False))

    if args.evaluate:
        print("\n--- Running pt.Experiment (BM25 vs TF-IDF) ---")
        eval_df = engine.evaluate()
        print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()
