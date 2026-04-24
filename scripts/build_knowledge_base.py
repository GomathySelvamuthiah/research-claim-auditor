"""
build_knowledge_base.py

Loads real paper abstracts from data/real_papers.json, embeds them with
sentence-transformers, and persists the FAISS-compatible index to data/cache/.

Usage:
    python scripts/build_knowledge_base.py
"""

import json
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.source_retriever import SourceRetriever

REAL_PAPERS_PATH = "data/real_papers.json"
INDEX_PATH = "data/cache/faiss_index"


def main() -> None:
    if not os.path.exists(REAL_PAPERS_PATH):
        print(f"❌  {REAL_PAPERS_PATH} not found — run scripts/fetch_real_data.py first.")
        sys.exit(1)

    with open(REAL_PAPERS_PATH) as f:
        data = json.load(f)
    papers = data.get("papers", [])

    print(f"Loaded {len(papers)} papers from {REAL_PAPERS_PATH}")
    print("Embedding and indexing (this may take ~30 s on first run)...\n")

    retriever = SourceRetriever(index_path=INDEX_PATH)
    retriever.clear()  # start fresh — avoids duplicate chunks on re-run

    n_papers = retriever.load_real_papers(REAL_PAPERS_PATH)
    n_chunks = len(retriever._chunks)

    retriever.save_index()

    print(f"Knowledge base built: {n_chunks} chunks from {n_papers} papers")
    print(f"Index saved to       : {INDEX_PATH}_chunks.json")
    print(f"Embeddings saved to  : {INDEX_PATH}_embeddings.npy")

    # Quick sanity-check retrieval
    test_query = "citation distortion academic papers"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"\nSanity check — top-3 results for '{test_query}':")
    for r in results:
        snippet = r.chunk.text[:80].replace("\n", " ")
        print(f"  [{r.score:.3f}] {r.chunk.title[:50]} — {snippet}…")


if __name__ == "__main__":
    main()
