"""Multimodal subset (table / figure / compute questions): text-only agent vs multimodal agent vs RAG."""

import sys

from run_benchmark import main

if __name__ == "__main__":
    names = ["hybrid_rerank_rag", "agent_text_only", "agent_heuristic"]
    sys.exit(main([f"configs/experiments/{n}.yaml" for n in names] + ["--types", "table,figure,compute,multihop",
                                                                      "--out", "benchmarks/results/multimodal",
                                                                      *sys.argv[1:]]))
