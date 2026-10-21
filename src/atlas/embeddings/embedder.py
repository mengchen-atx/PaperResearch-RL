"""Text -> vector.

An embedding model maps text to a point in R^d such that semantically similar
texts are close. We L2-normalize every vector, which makes cosine similarity
equal to the plain dot product:

    cos(q, d) = (q . d) / (|q| |d|)  ==  q . d   when |q| = |d| = 1

That identity is why the vector stores below can use inner-product search.

Two implementations:
  * SentenceTransformerEmbedder — real neural embeddings (default: bge-small-en-v1.5,
    384-d, strong quality for its size, runs on CPU/MPS).
  * HashingEmbedder — deterministic feature hashing of word n-grams. No model
    download; used for fast unit tests and as a lexical sanity baseline. It is NOT
    semantic and should never be used for reported results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

from atlas.common.config import EmbeddingConfig
from atlas.common.device import inference_lock, resolve_device
from atlas.common.ids import sha256_text
from atlas.observability.logging import get_logger

log = get_logger("embeddings")
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def name(self) -> str: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int(sha256_text(f)[:8], 16)
            v[h % self._dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        device = resolve_device(cfg.device)
        log.info("loading embedding model", extra={"model": cfg.model_name, "device": device})
        self._model = SentenceTransformer(cfg.model_name, device=device)
        self._dim = int((getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension)())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.cfg.model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        with inference_lock():
            vecs = self._model.encode(
                texts, batch_size=self.cfg.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # Asymmetric retrieval: bge expects an instruction prefix on queries only.
        return self.embed_documents([self.cfg.query_prefix + text])[0]


class CachedEmbedder:
    """Disk cache for document embeddings keyed by (model, sha256(text)).

    Re-ingesting an unchanged corpus (or re-running an ablation with a different
    retriever but the same embedder) skips the expensive encoder pass entirely.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", inner.name)
        self._path = cache_dir / f"emb_{safe}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self._path.exists():
            data = np.load(self._path)
            self._cache = {k: data[k] for k in data.files}

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def name(self) -> str:
        return self.inner.name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [sha256_text(t)[:32] for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            new = self.inner.embed_documents([texts[i] for i in missing])
            self._cache.update({keys[i]: new[j] for j, i in enumerate(missing)})
            self._persist()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, **self._cache)


def build_embedder(cfg: EmbeddingConfig, cache_dir: Path | None = None) -> Embedder:
    inner: Embedder
    inner = HashingEmbedder(cfg.hashing_dim) if cfg.provider == "hashing" else SentenceTransformerEmbedder(cfg)
    if cfg.cache and cache_dir is not None:
        return CachedEmbedder(inner, cache_dir)
    return inner
