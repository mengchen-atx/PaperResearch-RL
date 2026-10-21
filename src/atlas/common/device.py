"""Pick a torch device: CUDA > Apple MPS > CPU, unless explicitly configured."""

from __future__ import annotations

import threading


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Local model inference (embedder, reranker, LLM, VLM) can be invoked from different
# threads: tool executor workers, API request threads, the agent's main thread. PyTorch's
# MPS backend is not thread-safe (observed: Metal "command encoder is already encoding"
# abort when the VLM and another model ran concurrently), and HF `generate` on one model
# instance is not re-entrant either. A single process-wide re-entrant lock serializes
# local inference; concurrency comes from running more processes/replicas, or from vLLM,
# which batches requests internally.
_INFERENCE_LOCK = threading.RLock()


def inference_lock() -> threading.RLock:
    return _INFERENCE_LOCK
