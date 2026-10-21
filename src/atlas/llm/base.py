"""Provider-agnostic LLM interface.

Every provider implements `_generate` (and optionally `_stream`). The base class
adds what every call needs regardless of provider: tracing spans, Prometheus
token/latency metrics, retries with exponential backoff, and a deterministic
response cache (only at temperature 0, where outputs are reproducible).

Providers (see `atlas.llm.providers`):
  hf_local           transformers on CPU/MPS/CUDA (+ optional LoRA adapter)
  openai_compatible  any /v1/chat/completions server: vLLM, Ollama, OpenAI, the
                     bundled dev server (`atlas serve-model`)
  anthropic          Anthropic Messages API
  extractive         no-model baseline that quotes the best-matching evidence
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from atlas.common.cache import Cache, NullCache, make_key
from atlas.common.errors import LLMError
from atlas.common.types import TokenUsage
from atlas.observability.logging import get_logger
from atlas.observability.metrics import LLM_LATENCY, LLM_TOKENS
from atlas.observability.tracing import get_tracer

log = get_logger("llm")


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    finish_reason: str | None = None
    cached: bool = False


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


class LLMClient:
    """Template-method base class. Subclasses implement `_generate`."""

    provider: str = "base"

    def __init__(self, model: str, max_retries: int = 2, cache: Cache | None = None,
                 price_in: float = 0.0, price_out: float = 0.0) -> None:
        self.model = model
        self.max_retries = max_retries
        self.cache = cache or NullCache()
        self.price_in = price_in
        self.price_out = price_out

    # --- subclass hooks -----------------------------------------------------------
    def _generate(self, messages: list[Message], max_tokens: int, temperature: float) -> LLMResponse:
        raise NotImplementedError

    def _stream(self, messages: list[Message], max_tokens: int, temperature: float) -> Iterator[str]:
        yield self._generate(messages, max_tokens, temperature).text

    # --- public API -----------------------------------------------------------------
    def generate(self, messages: list[Message], max_tokens: int = 384, temperature: float = 0.0,
                 purpose: str = "generate") -> LLMResponse:
        key = None
        if temperature == 0.0:
            key = make_key("llm", {"m": self.model, "p": self.provider, "t": max_tokens,
                                   "msgs": [m.model_dump() for m in messages]})
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResponse.model_validate({**hit, "cached": True})

        with get_tracer().span(f"llm.{purpose}", kind="llm", model=self.model, provider=self.provider) as attrs:
            resp = self._with_retries(messages, max_tokens, temperature)
            attrs.update(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
                         latency_ms=round(resp.latency_ms, 1))
        LLM_TOKENS.labels(self.model, "prompt").inc(resp.usage.prompt_tokens)
        LLM_TOKENS.labels(self.model, "completion").inc(resp.usage.completion_tokens)
        LLM_LATENCY.labels(self.model).observe(resp.latency_ms / 1000)
        if key is not None:
            self.cache.set(key, resp.model_dump(mode="json"))
        return resp

    def stream(self, messages: list[Message], max_tokens: int = 384, temperature: float = 0.0) -> Iterator[str]:
        yield from self._stream(messages, max_tokens, temperature)

    def cost_usd(self, usage: TokenUsage) -> float:
        return (usage.prompt_tokens * self.price_in + usage.completion_tokens * self.price_out) / 1e6

    def _with_retries(self, messages: list[Message], max_tokens: int, temperature: float) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._generate(messages, max_tokens, temperature)
                if not resp.latency_ms:
                    resp = resp.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})
                return resp
            except LLMError as exc:  # non-retryable (bad request, auth, invalid output)
                raise exc
            except Exception as exc:  # network errors, timeouts, 5xx
                last = exc
                wait = min(8.0, 0.5 * 2**attempt)
                log.warning("llm call failed; retrying", extra={"attempt": attempt + 1, "wait_s": wait,
                                                                "error": f"{type(exc).__name__}: {exc}"})
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise LLMError(f"LLM call to {self.provider}:{self.model} failed after retries: {last}") from last
