"""The on-disk knowledge index: chunks + dense vectors + BM25 postings.

Layout of `index_dir/`:
    chunks.jsonl     one Chunk per line (row i == vector i == BM25 doc i)
    documents.jsonl  one Document per line
    vectors.npy | faiss.index | qdrant.json
    bm25.json
    manifest.json    embedding model, dim, store type, counts, build time
    tables.sqlite    structured tables for the SQL tool (multimodal)
    figures/         rendered figure PNGs (multimodal)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from atlas.common.config import AtlasConfig
from atlas.common.errors import IndexNotFoundError
from atlas.common.types import Chunk, Document
from atlas.embeddings.embedder import Embedder
from atlas.retrieval.bm25 import BM25Index
from atlas.retrieval.vector_store import FaissVectorStore, NumpyVectorStore, QdrantVectorStore, VectorStore


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    documents: list[Document]
    vector_store: VectorStore
    bm25: BM25Index
    manifest: dict

    def chunk_by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}


def _make_store(cfg: AtlasConfig, dim: int) -> VectorStore:
    kind = cfg.retrieval.vector_store
    if kind == "numpy":
        return NumpyVectorStore(dim)
    if kind == "faiss":
        return FaissVectorStore(dim)
    r = cfg.retrieval
    return QdrantVectorStore(dim, r.qdrant_collection, url=r.qdrant_url, path=str(cfg.resolve_path(r.qdrant_path)))


def build_index(
    cfg: AtlasConfig, embedder: Embedder, documents: list[Document], chunks: list[Chunk]
) -> KnowledgeIndex:
    vectors = embedder.embed_documents([c.text for c in chunks])
    store = _make_store(cfg, embedder.dim)
    if isinstance(store, QdrantVectorStore):
        store.recreate()
        store.add(vectors, chunks)
    else:
        store.add(vectors)
    bm25 = BM25Index(cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b).build([c.text for c in chunks])
    manifest = {
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "vector_store": store.name,
        "num_chunks": len(chunks),
        "num_documents": len(documents),
        "chunking": cfg.chunking.model_dump(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return KnowledgeIndex(chunks, documents, store, bm25, manifest)


def save_index(index: KnowledgeIndex, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w") as f:
        for c in index.chunks:
            f.write(c.model_dump_json() + "\n")
    with (directory / "documents.jsonl").open("w") as f:
        for d in index.documents:
            f.write(d.model_dump_json() + "\n")
    index.vector_store.save(directory)
    (directory / "bm25.json").write_text(json.dumps(index.bm25.to_dict()))
    (directory / "manifest.json").write_text(json.dumps(index.manifest, indent=2))


def load_index(cfg: AtlasConfig, directory: Path) -> KnowledgeIndex:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise IndexNotFoundError(str(directory))
    manifest = json.loads(manifest_path.read_text())
    chunks = [Chunk.model_validate_json(line) for line in (directory / "chunks.jsonl").read_text().splitlines() if line]
    documents = [
        Document.model_validate_json(line) for line in (directory / "documents.jsonl").read_text().splitlines() if line
    ]
    dim = int(manifest["dim"])
    kind = manifest["vector_store"]
    store: VectorStore
    if kind == "numpy":
        store = NumpyVectorStore.load(directory, dim)
    elif kind == "faiss":
        store = FaissVectorStore.load(directory, dim)
    else:
        r = cfg.retrieval
        store = QdrantVectorStore(dim, r.qdrant_collection, url=r.qdrant_url, path=str(cfg.resolve_path(r.qdrant_path)))
    bm25 = BM25Index.from_dict(json.loads((directory / "bm25.json").read_text()))
    if len(store) != len(chunks):
        raise IndexNotFoundError(f"{directory} (corrupt: {len(store)} vectors vs {len(chunks)} chunks)")
    return KnowledgeIndex(chunks, documents, store, bm25, manifest)

