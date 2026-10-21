"""Metadata filtering: restrict retrieval to a subset of the corpus.

Examples: "only tables", "only attention-paper pages 3-5". Filtering happens
*before* scoring (pre-filtering), so Top-K is computed within the allowed set
rather than filtering an already-truncated list (which can return < K results).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from atlas.common.types import Chunk, Modality


class MetadataFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_ids: list[str] | None = None
    filenames: list[str] | None = None
    modalities: list[Modality] | None = None
    page_min: int | None = None
    page_max: int | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    def matches(self, chunk: Chunk) -> bool:
        if self.doc_ids is not None and chunk.doc_id not in self.doc_ids:
            return False
        if self.filenames is not None and chunk.filename not in self.filenames:
            return False
        if self.modalities is not None and chunk.modality not in self.modalities:
            return False
        if self.page_min is not None and (chunk.page is None or chunk.page < self.page_min):
            return False
        return not (self.page_max is not None and (chunk.page is None or chunk.page > self.page_max))


def allowed_indices(chunks: list[Chunk], flt: MetadataFilter | None) -> np.ndarray | None:
    """Indices of chunks passing the filter, or None meaning 'all chunks'."""
    if flt is None or flt.is_empty():
        return None
    return np.array([i for i, c in enumerate(chunks) if flt.matches(c)], dtype=np.int64)
