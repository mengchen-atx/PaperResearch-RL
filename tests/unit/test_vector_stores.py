import numpy as np
import pytest

from atlas.common.types import Chunk, Modality
from atlas.embeddings.embedder import HashingEmbedder, l2_normalize
from atlas.retrieval.filters import MetadataFilter, allowed_indices
from atlas.retrieval.vector_store import (
    FaissVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    top_k_from_scores,
)


@pytest.fixture
def vectors():
    rng = np.random.default_rng(0)
    return l2_normalize(rng.normal(size=(50, 16)).astype(np.float32))


def test_top_k_from_scores_sorted_and_filtered():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert [i for i, _ in top_k_from_scores(s, 2)] == [1, 3]
    assert [i for i, _ in top_k_from_scores(s, 5, allowed=np.array([0, 2]))] == [2, 0]
    assert top_k_from_scores(s, 3, allowed=np.array([], dtype=np.int64)) == []


def test_numpy_faiss_qdrant_agree(vectors):
    q = vectors[7]
    stores = [NumpyVectorStore(16), FaissVectorStore(16), QdrantVectorStore(16, "t", path=None)]
    stores[2].recreate()
    results = []
    for st in stores:
        st.add(vectors)
        results.append([i for i, _ in st.search(q, 5)])
    assert results[0][0] == 7  # a vector is most similar to itself (cosine == 1)
    assert results[0] == results[1] == results[2]


def test_filtered_search_consistent(vectors):
    allowed = np.array([3, 4, 5], dtype=np.int64)
    np_res = [i for i, _ in NumpyVectorStore(16, vectors).search(vectors[0], 2, allowed=allowed)]
    fs = FaissVectorStore(16)
    fs.add(vectors)
    assert np_res == [i for i, _ in fs.search(vectors[0], 2, allowed=allowed)]
    assert set(np_res) <= {3, 4, 5}


def test_hashing_embedder_normalized_and_deterministic():
    e = HashingEmbedder(64)
    a, b = e.embed_documents(["hello world", "hello world"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    assert e.embed_documents([]).shape == (0, 64)


def test_metadata_filter():
    chunks = [Chunk(chunk_id=str(i), doc_id="d", filename=f"f{i % 2}.pdf", text="x", page=i,
                    modality=Modality.TABLE if i == 3 else Modality.TEXT) for i in range(5)]
    assert allowed_indices(chunks, None) is None
    assert allowed_indices(chunks, MetadataFilter(modalities=[Modality.TABLE])).tolist() == [3]
    assert allowed_indices(chunks, MetadataFilter(filenames=["f0.pdf"], page_min=1)).tolist() == [2, 4]
