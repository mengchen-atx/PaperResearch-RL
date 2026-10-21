# Design: Retrieval (V0 handwritten RAG → V1 advanced RAG)

Read in this order: `ingestion/loaders.py`, `chunking/chunker.py`, `embeddings/embedder.py`,
`retrieval/vector_store.py`, `retrieval/bm25.py`, `retrieval/fusion.py`, `reranking/reranker.py`,
`retrieval/retriever.py`, `rag/prompts.py`, `rag/pipeline.py`, `rag/citations.py`.

| Decision | Choice | Alternatives | Why |
|---|---|---|---|
| Parser | PyMuPDF text layer | OCR, layout models | born-digital PDFs; exact + fast |
| Chunk unit | sentences packed to 1000 chars, 150 overlap, page-bounded | fixed windows, semantic chunks | precise citations, intact facts |
| Embedder | bge-small-en-v1.5 (384-d) | MiniLM, e5, OpenAI embeddings | quality/size, local |
| Similarity | cosine via normalized dot product | L2, learned metrics | standard for sentence embeddings |
| Index | FAISS FlatIP (exact); NumPy reference; Qdrant adapter | HNSW/IVF | exact is fast at this scale |
| Sparse | hand-written Okapi BM25 (k1=1.5, b=0.75) | SPLADE, Elasticsearch | exact-term matching, zero deps |
| Fusion | RRF (k=60); weighted min-max | learned fusion | scale-free, no tuning |
| Rerank | cross-encoder MiniLM-L6 over top-30 | ColBERT, LLM rerankers | precision per ms on CPU |
| Prompt | numbered evidence, cite `[n]`, source-paper refs rewritten to `(ref n)` | free-form | measurable citations |

**Citation hygiene:** source papers contain bibliography markers like `[34, 2, 9]`; small models copied them
verbatim as if they were our citations (observed in the first end-to-end run and flagged by the validator), so
the context rewrites them to `(ref 34, 2, 9)`.

**Empty retrieval:** the pipeline refuses to generate an ungrounded answer and says no evidence was found.

**Query rewriting:** `keywords` (stopword-stripped) and `llm` modes exist; reranking always scores against the
original question.
