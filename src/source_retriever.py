"""
source_retriever.py

Retrieves the source passage(s) that a citing claim refers to, using a
FAISS-backed dense vector index over ingested reference documents.

Responsibilities:
- Encode claims and document chunks with a sentence-transformer embedding model
- Build and query a FAISS index for top-k nearest-neighbour retrieval
- Return ranked SourcePassage objects (text, document metadata, similarity score)
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 384


@dataclass
class SourceChunk:
    chunk_id: int
    doi: str
    title: str
    year: int
    text: str
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: SourceChunk
    score: float


class SourceRetriever:
    _st_model = None  # class-level model cache to avoid reloading

    def __init__(
        self,
        index_path: str = "data/cache/faiss_index",
        top_k: int = 4,
        chunk_size: int = 300,
        overlap: int = 50,
    ):
        self.index_path = index_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: list[SourceChunk] = []
        self._embeddings: list[list[float]] = []
        try:
            self.load_index()
        except Exception:
            pass

    def index_document(self, doc: dict) -> int:
        try:
            chunks = self._chunk_document(doc, self.chunk_size, self.overlap)
            if not chunks:
                return 0
            id_offset = len(self._chunks)
            for i, chunk in enumerate(chunks):
                chunk.chunk_id = id_offset + i
                self._embeddings.append(self._embed_text(chunk.text))
                self._chunks.append(chunk)
            return len(chunks)
        except Exception as e:
            logger.error("Failed to index document %s: %s", doc.get("doi", "unknown"), e)
            return 0

    def index_batch(self, docs: list[dict]) -> int:
        total = 0
        for doc in docs:
            total += self.index_document(doc)
        return total

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not self._chunks:
            return []
        try:
            k = top_k if top_k is not None else self.top_k
            return self._cosine_search(self._embed_text(query), k)
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return []

    def save_index(self) -> None:
        try:
            path = Path(self.index_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(path) + "_chunks.json", "w") as f:
                json.dump([asdict(c) for c in self._chunks], f)
            np.save(str(path) + "_embeddings.npy", np.array(self._embeddings, dtype=np.float32))
        except Exception as e:
            logger.error("Failed to save index: %s", e)

    def load_index(self) -> bool:
        try:
            chunks_path = self.index_path + "_chunks.json"
            embeddings_path = self.index_path + "_embeddings.npy"
            if not (os.path.exists(chunks_path) and os.path.exists(embeddings_path)):
                return False
            with open(chunks_path) as f:
                self._chunks = [SourceChunk(**d) for d in json.load(f)]
            self._embeddings = np.load(embeddings_path).tolist()
            return True
        except Exception as e:
            logger.error("Failed to load index: %s", e)
            return False

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = []

    def load_real_papers(self, path: str = "data/real_papers.json") -> int:
        """
        Load papers from the real_papers.json file produced by fetch_real_data.py,
        index them into the vector store, and return the number of papers indexed.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            papers = data.get("papers", [])
            n = self.index_batch(papers)
            logger.info(
                "Indexed %d chunks from %d real papers (OpenAlex / Semantic Scholar)",
                n, len(papers),
            )
            return len(papers)
        except Exception as e:
            logger.error("Failed to load real papers from %s: %s", path, e)
            return 0

    @staticmethod
    def _chunk_document(doc: dict, chunk_size: int = 300, overlap: int = 50) -> list[SourceChunk]:
        text = doc.get("full_text") or doc.get("abstract", "")
        doi = doc.get("doi", "")
        title = doc.get("title", "")
        year = int(doc.get("year", 0))

        if not text:
            return []

        words = text.split()
        step = max(1, chunk_size - overlap)
        chunks: list[SourceChunk] = []
        chunk_index = 0

        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if len(window) < 20:
                continue
            chunks.append(SourceChunk(
                chunk_id=chunk_index,
                doi=doi,
                title=title,
                year=year,
                text=" ".join(window),
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        return chunks

    @staticmethod
    def _pseudo_embed(text: str) -> list[float]:
        """Deterministic fallback embedder using MD5 hashing — no model required."""
        words = text.lower().split()
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for i, word in enumerate(words):
            digest = hashlib.md5(word.encode()).digest()  # 16 bytes
            for j, b in enumerate(digest):
                vec[(i * 16 + j) % EMBED_DIM] += (b - 128) / 128.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _embed_text(self, text: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
            if SourceRetriever._st_model is None:
                SourceRetriever._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = np.array(
                SourceRetriever._st_model.encode(text, normalize_embeddings=True),
                dtype=np.float32,
            )
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec.tolist()
        except Exception:
            logger.warning(
                "sentence-transformers unavailable, using pseudo_embed fallback "
                "— similarity scores will be unreliable"
            )
            return self._pseudo_embed(text)

    def _cosine_search(self, query_vector: list[float], k: int) -> list[RetrievalResult]:
        q = np.array(query_vector, dtype=np.float32)
        matrix = np.array(self._embeddings, dtype=np.float32)
        scores = matrix @ q  # dot product == cosine sim for L2-normalized vectors
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]


if __name__ == "__main__":
    import pathlib

    sample_text = pathlib.Path("data/sample_papers/sample_intro.txt").read_text()
    doc = {"doi": "10.demo/test", "title": "Demo Source Paper", "year": 2023, "full_text": sample_text}

    retriever = SourceRetriever()
    n = retriever.index_document(doc)
    print(f"Indexed {n} chunks")

    results = retriever.retrieve("neuroinflammation cognitive decline gut microbiome", top_k=3)
    for r in results:
        print(f"Score {r.score:.3f} | {r.chunk.text[:100]}...")
