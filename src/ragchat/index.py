"""Hybrid search index: dense embeddings + BM25, merged with reciprocal rank fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .documents import Chunk
from .llm import Embedder
from .text import tokenize

# Standard RRF constant; it damps the gap between rank 1 and rank 2.
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None):
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("one vector per chunk is required")
        self.chunks = chunks
        self.vectors = _normalize(vectors) if vectors is not None else None
        # BM25Okapi cannot be built from an empty corpus.
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None) -> "Index":
        vectors = embedder.embed_documents([c.text for c in chunks]) if embedder and chunks else None
        return cls(chunks, vectors)

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None

    def search(self, query: str, k: int, embedder: Embedder | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        depth = min(len(self.chunks), max(k * 4, 20))
        rankings = [self._keyword_ranking(query, depth)]
        if self.has_vectors and embedder is not None:
            rankings.append(self._dense_ranking(embedder.embed_query(query), depth))

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(self.chunks[i], score) for i, score in best]

    def _keyword_ranking(self, query: str, depth: int) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)[:depth]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_ranking(self, query_vector: np.ndarray, depth: int) -> list[int]:
        q = query_vector / (np.linalg.norm(query_vector) or 1.0)
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:depth]]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        if self.vectors is not None:
            np.save(folder / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, folder: str | Path) -> "Index":
        folder = Path(folder)
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        vectors_path = folder / "vectors.npy"
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        return cls(chunks, vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
