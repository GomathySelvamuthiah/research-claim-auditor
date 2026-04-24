"""Tests for src/source_retriever.py"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.source_retriever import EMBED_DIM, RetrievalResult, SourceChunk, SourceRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(word_count: int = 400, doi: str = "10.test/001") -> dict:
    """Build a minimal doc dict with a full_text of exactly word_count words."""
    return {
        "doi": doi,
        "title": "Test Paper",
        "year": 2021,
        "full_text": " ".join(["word"] * word_count),
    }


def _retriever_with_pseudo() -> SourceRetriever:
    """Fresh retriever whose _embed_text is replaced with _pseudo_embed."""
    r = SourceRetriever()
    r.clear()
    r._embed_text = SourceRetriever._pseudo_embed
    return r


# ---------------------------------------------------------------------------
# _chunk_document
# ---------------------------------------------------------------------------

def test_chunk_document_produces_chunks():
    doc = _make_doc(word_count=200)
    chunks = SourceRetriever._chunk_document(doc, chunk_size=300, overlap=50)
    assert len(chunks) >= 1


def test_chunk_document_sets_doi():
    doc = _make_doc(word_count=400, doi="10.example/42")
    chunks = SourceRetriever._chunk_document(doc, chunk_size=300, overlap=50)
    assert all(c.doi == "10.example/42" for c in chunks)


def test_chunk_document_skips_short_text():
    doc = _make_doc(word_count=10)
    chunks = SourceRetriever._chunk_document(doc, chunk_size=300, overlap=50)
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# _pseudo_embed
# ---------------------------------------------------------------------------

def test_pseudo_embed_deterministic():
    text = "neuroinflammation drives cognitive decline in diabetic patients"
    vec1 = SourceRetriever._pseudo_embed(text)
    vec2 = SourceRetriever._pseudo_embed(text)
    assert vec1 == vec2


def test_pseudo_embed_length():
    vec = SourceRetriever._pseudo_embed("hello world test sentence")
    assert len(vec) == EMBED_DIM


def test_pseudo_embed_normalized():
    vec = SourceRetriever._pseudo_embed("gut microbiome dysbiosis LPS endotoxemia")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# index + retrieve (pseudo-embed only, no model loading)
# ---------------------------------------------------------------------------

def test_index_and_retrieve():
    retriever = _retriever_with_pseudo()
    doc = _make_doc(word_count=600, doi="10.test/retrieve")
    n = retriever.index_document(doc)
    assert n > 0

    results = retriever.retrieve("word", top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.chunk.doi == "10.test/retrieve" for r in results)
    # Scores should be in descending order
    if len(results) == 2:
        assert results[0].score >= results[1].score


def test_retrieve_empty_index_returns_empty():
    retriever = SourceRetriever()
    retriever.clear()
    retriever._embed_text = SourceRetriever._pseudo_embed
    results = retriever.retrieve("any query")
    assert results == []
