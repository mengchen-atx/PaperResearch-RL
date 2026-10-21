"""OpenAI-compatible inference server backed by HF transformers (development stand-in for vLLM).

Exposes the same `/v1/chat/completions` contract (incl. SSE streaming) as vLLM,
so the application's `openai_compatible` provider, the inference benchmark
(`scripts/bench_inference.py`) and the deployment manifests are exercised end to
end on machines without an NVIDIA GPU. It processes one request at a time — it
has none of vLLM's continuous batching or PagedAttention, which is exactly the
difference the inference benchmark is designed to expose when run against vLLM.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from atlas.common.config import LLMConfig
from atlas.llm.base import Message
from atlas.llm.providers import HFLocalLLM


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = 0.0
    stream: bool = False


def create_model_app(cfg: LLMConfig) -> FastAPI:
    llm = HFLocalLLM(cfg.model_copy(update={"provider": "hf_local"}))
    app = FastAPI(title="Atlas dev model server (OpenAI-compatible)")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": llm.model}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": llm.model, "object": "model", "owned_by": "atlas"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        msgs = [Message(role=m.role, content=m.content) for m in req.messages]  # type: ignore[arg-type]
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if req.stream:
            def gen():
                for delta in llm.stream(msgs, max_tokens=req.max_tokens, temperature=req.temperature):
                    chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": llm.model, "choices": [{"index": 0, "delta": {"content": delta}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        resp = llm.generate(msgs, max_tokens=req.max_tokens, temperature=req.temperature)
        return {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": llm.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.text},
                             "finish_reason": resp.finish_reason}],
                "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                          "total_tokens": resp.usage.total_tokens}}

    return app
