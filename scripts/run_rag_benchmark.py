"""Retrieval + RAG ablation ladder: retrieval variants, then base LLM -> vanilla -> hybrid -> hybrid+rerank."""

import sys

from run_benchmark import main

RETRIEVAL = ["retrieval_dense", "retrieval_bm25", "retrieval_hybrid", "retrieval_hybrid_weighted",
             "retrieval_dense_rerank", "retrieval_hybrid_rerank"]
GENERATION = ["extractive_rag", "base_llm", "vanilla_rag", "hybrid_rag", "hybrid_rerank_rag"]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = RETRIEVAL if which == "retrieval" else GENERATION if which == "generation" else RETRIEVAL + GENERATION
    sys.exit(main([f"configs/experiments/{n}.yaml" for n in names] + sys.argv[2:]))
