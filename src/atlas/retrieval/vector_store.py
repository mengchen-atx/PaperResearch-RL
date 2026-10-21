"""Vector stores: exact brute-force (NumPy), FAISS, and Qdrant.

All stores hold L2-normalized vectors and score by inner product (= cosine).
Row i of the store corresponds to chunk i of the index's chunk list.

  * NumpyVectorStore — the textbook implementation: scores = V @ q, then partial
    sort. O(N*d) per query; perfectly fine up to ~1e5 chunks and easy to read.
  * FaissVectorStore — `IndexFlatIP`: same exact math in optimized C++/SIMD. For
    millions of vectors swap in an approximate index (IVF/HNSW) — the interface
    stays the same.
  * QdrantVectorStore — a production vector database adapter (persistent,
    networked, native payload filtering, horizontal scaling). Runs embedded
    (`path=`) locally or against a server (`url=`) in the distributed deployment.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from atlas.common.types import Chunk
from atlas.retrieval.filters import MetadataFilter


def import_faiss():
    """Import faiss safely.

    On macOS, faiss-cpu and torch wheels each ship an OpenMP runtime. If faiss
    loads first, torch's later OpenMP use segfaults the process (reproduced:
    faiss search -> load Qwen2.5 -> SIGSEGV). Importing torch first makes both
    share torch's runtime. No-op cost on Linux beyond the torch import.
    """
    if sys.platform == "darwin":
        with contextlib.suppress(ImportError):
            import torch  # noqa: F401
    import faiss

    return faiss


class VectorStore(Protocol):
    name: str

    def add(self, vectors: np.ndarray) -> None: ...
    def search(
        self, query: np.ndarray, k: int, allowed: np.ndarray | None = None, flt: MetadataFilter | None = None
    ) -> list[tuple[int, float]]: ...
    def save(self, directory: Path) -> None: ...
    def __len__(self) -> int: ...


def top_k_from_scores(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Exact top-k by score. `argpartition` is O(N) vs O(N log N) for a full sort."""
    if allowed is not None:
        if allowed.size == 0:
            return []
        sub = scores[allowed]
        k = min(k, sub.size)
        part = np.argpartition(-sub, k - 1)[:k]
        order = part[np.argsort(-sub[part])]
        return [(int(allowed[i]), float(sub[i])) for i in order]
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    return [(int(i), float(scores[i])) for i in order]


class NumpyVectorStore:
    name = "numpy"

    def __init__(self, dim: int, vectors: np.ndarray | None = None) -> None:
        self.dim = dim
        self._vectors = vectors if vectors is not None else np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query, k, allowed=None, flt=None):
        scores = self._vectors @ query.astype(np.float32)  # cosine similarity for unit vectors
        return top_k_from_scores(scores, k, allowed)

    def save(self, directory: Path) -> None:
        np.save(directory / "vectors.npy", self._vectors)

    @classmethod
    def load(cls, directory: Path, dim: int) -> NumpyVectorStore:
        return cls(dim, np.load(directory / "vectors.npy"))

    def __len__(self) -> int:
        return int(self._vectors.shape[0])


class FaissVectorStore:
    name = "faiss"

    def __init__(self, dim: int, index=None) -> None:
        faiss = import_faiss()
        self._faiss = faiss
        self.dim = dim
        self._index = index if index is not None else faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query, k, allowed=None, flt=None):
        if len(self) == 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        params = None
        if allowed is not None:
            if allowed.size == 0:
                return []
            params = self._faiss.SearchParameters(sel=self._faiss.IDSelectorBatch(allowed.astype(np.int64)))
            k = min(k, int(allowed.size))
        k = min(k, len(self))
        scores, ids = self._index.search(q, k, params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def save(self, directory: Path) -> None:
        self._faiss.write_index(self._index, str(directory / "faiss.index"))

    @classmethod
    def load(cls, directory: Path, dim: int) -> FaissVectorStore:
        faiss = import_faiss()
        return cls(dim, faiss.read_index(str(directory / "faiss.index")))

    def __len__(self) -> int:
        return int(self._index.ntotal)


class QdrantVectorStore:
    """Qdrant adapter. Point id == chunk row index; payload carries filterable metadata."""

    name = "qdrant"

    def __init__(self, dim: int, collection: str, url: str | None = None, path: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.dim = dim
        self.collection = collection
        self._client = QdrantClient(url=url) if url else QdrantClient(path=path or ":memory:")
        self._count = 0
        if self._client.collection_exists(collection):
            self._count = self._client.count(collection).count

    def recreate(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
        self._count = 0

    def add(self, vectors: np.ndarray, chunks: list[Chunk] | None = None) -> None:
        from qdrant_client.models import PointStruct

        if not self._client.collection_exists(self.collection):
            self.recreate()
        points = []
        for j, vec in enumerate(vectors):
            idx = self._count + j
            c = chunks[j] if chunks else None
            payload = (
                {"doc_id": c.doc_id, "filename": c.filename, "modality": str(c.modality), "page": c.page}
                if c else {}
            )
            points.append(PointStruct(id=idx, vector=vec.tolist(), payload=payload))
        for start in range(0, len(points), 256):
            self._client.upsert(self.collection, points=points[start : start + 256])
        self._count += len(points)

    def _to_qdrant_filter(self, flt: MetadataFilter | None, allowed: np.ndarray | None):
        from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchAny, Range

        must = []
        if flt is not None and not flt.is_empty():
            if flt.doc_ids:
                must.append(FieldCondition(key="doc_id", match=MatchAny(any=flt.doc_ids)))
            if flt.filenames:
                must.append(FieldCondition(key="filename", match=MatchAny(any=flt.filenames)))
            if flt.modalities:
                must.append(FieldCondition(key="modality", match=MatchAny(any=[str(m) for m in flt.modalities])))
            if flt.page_min is not None or flt.page_max is not None:
                must.append(FieldCondition(key="page", range=Range(gte=flt.page_min, lte=flt.page_max)))
        elif allowed is not None:
            must.append(HasIdCondition(has_id=[int(i) for i in allowed]))
        return Filter(must=must) if must else None

    def search(self, query, k, allowed=None, flt=None):
        if self._count == 0 or (allowed is not None and allowed.size == 0):
            return []
        res = self._client.query_points(
            self.collection, query=query.tolist(), limit=k, query_filter=self._to_qdrant_filter(flt, allowed)
        )
        return [(int(p.id), float(p.score)) for p in res.points]

    def save(self, directory: Path) -> None:
        (directory / "qdrant.json").write_text(json.dumps({"collection": self.collection, "count": self._count}))

    def close(self) -> None:
        self._client.close()

    def __len__(self) -> int:
        return self._count
