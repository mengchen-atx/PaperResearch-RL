"""Deterministic identifiers.

Chunk IDs are content-addressed (doc + position + text hash) so re-ingesting the
same corpus yields the same IDs. That stability is what lets benchmark labels,
caches, and trajectories reference chunks across runs.
"""

from __future__ import annotations

import hashlib
import uuid


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for(filename: str, content_hash: str) -> str:
    stem = "".join(ch if ch.isalnum() else "_" for ch in filename.rsplit(".", 1)[0]).strip("_")
    return f"{stem[:48]}-{content_hash[:8]}"


def chunk_id_for(doc_id: str, kind: str, index: int, text: str) -> str:
    return f"{doc_id}:{kind}{index:04d}:{sha256_text(text)[:8]}"


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"
