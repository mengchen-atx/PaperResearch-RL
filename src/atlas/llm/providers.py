"""Concrete LLM providers."""

from __future__ import annotations

import json
import re
import threading
import time

import httpx

from atlas.common.cache import Cache
from atlas.common.config import LLMConfig, read_secret
from atlas.common.device import inference_lock, resolve_device
from atlas.common.errors import ConfigError, LLMError
from atlas.common.types import TokenUsage
from atlas.llm.base import LLMClient, LLMResponse, Message
from atlas.observability.logging import get_logger
from atlas.retrieval.bm25 import tokenize

log = get_logger("llm.providers")


# --------------------------------------------------------------------------------------
# Local Hugging Face model
# --------------------------------------------------------------------------------------
class HFLocalLLM(LLMClient):
    """Runs an open-weights chat model with transformers (optionally + LoRA adapter).

    Intended for development and for evaluating fine-tuned adapters locally. For
    concurrent production traffic use vLLM (continuous batching, paged KV cache)
    through `OpenAICompatibleLLM`.
    """

    provider = "hf_local"

    def __init__(self, cfg: LLMConfig, cache: Cache | None = None) -> None:
        super().__init__(cfg.model, cfg.max_retries, cache, cfg.price_input_per_mtok, cfg.price_output_per_mtok)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(cfg.device)
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device, torch.float32)
        log.info("loading local LLM", extra={"model": cfg.model, "device": self.device, "adapter": cfg.adapter_path})
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
        if cfg.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg.adapter_path)
            self.model = f"{cfg.model}+lora:{cfg.adapter_path}"
        self._model = model.to(self.device).eval()
        self._lock = inference_lock()  # process-wide: MPS and HF generate are not thread-safe

    def _inputs(self, messages: list[Message]):
        prompt = self.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def _gen_kwargs(self, max_tokens: int, temperature: float) -> dict:
        kw = {"max_new_tokens": max_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return kw

    def _generate(self, messages, max_tokens, temperature):
        import torch

        inputs = self._inputs(messages)
        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            out = self._model.generate(**inputs, **self._gen_kwargs(max_tokens, temperature))
        n_in = inputs["input_ids"].shape[1]
        new_tokens = out[0][n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return LLMResponse(
            text=text.strip(), model=self.model,
            usage=TokenUsage(prompt_tokens=int(n_in), completion_tokens=int(new_tokens.shape[0])),
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason="length" if new_tokens.shape[0] >= max_tokens else "stop",
        )

    def _stream(self, messages, max_tokens, temperature):
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            with self._lock:
                self._model.generate(**inputs, streamer=streamer, **self._gen_kwargs(max_tokens, temperature))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield from streamer
        thread.join()


# --------------------------------------------------------------------------------------
# OpenAI-compatible HTTP (vLLM / Ollama / OpenAI / atlas dev server)
# --------------------------------------------------------------------------------------
class OpenAICompatibleLLM(LLMClient):
    provider = "openai_compatible"

    def __init__(self, cfg: LLMConfig, cache: Cache | None = None, transport: httpx.BaseTransport | None = None):
        super().__init__(cfg.model, cfg.max_retries, cache, cfg.price_input_per_mtok, cfg.price_output_per_mtok)
        if not cfg.base_url:
            raise ConfigError("llm.base_url is required for provider 'openai_compatible' (e.g. http://localhost:8001/v1)")
        headers = {"Content-Type": "application/json"}
        key = read_secret(cfg.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(base_url=cfg.base_url.rstrip("/"), headers=headers,
                                    timeout=cfg.timeout_s, transport=transport)

    def _payload(self, messages, max_tokens, temperature, stream=False) -> dict:
        return {"model": self.model, "messages": [m.model_dump() for m in messages],
                "max_tokens": max_tokens, "temperature": temperature, "stream": stream}

    def _generate(self, messages, max_tokens, temperature):
        t0 = time.perf_counter()
        r = self._client.post("/chat/completions", json=self._payload(messages, max_tokens, temperature))
        if 400 <= r.status_code < 500:
            raise LLMError(f"LLM request rejected ({r.status_code}): {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice["message"].get("content") or "").strip(), model=data.get("model", self.model),
            usage=TokenUsage(prompt_tokens=usage.get("prompt_tokens", 0),
                             completion_tokens=usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000, finish_reason=choice.get("finish_reason"),
        )

    def _stream(self, messages, max_tokens, temperature):
        payload = self._payload(messages, max_tokens, temperature, stream=True)
        with self._client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


# --------------------------------------------------------------------------------------
# Anthropic Messages API (raw HTTP; no SDK dependency)
# --------------------------------------------------------------------------------------
class AnthropicLLM(LLMClient):
    provider = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, cfg: LLMConfig, cache: Cache | None = None, transport: httpx.BaseTransport | None = None):
        super().__init__(cfg.model, cfg.max_retries, cache, cfg.price_input_per_mtok, cfg.price_output_per_mtok)
        key = read_secret(cfg.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError(f"Set {cfg.api_key_env or 'ANTHROPIC_API_KEY'} to use the anthropic provider")
        self._client = httpx.Client(
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout=cfg.timeout_s, transport=transport,
        )

    def _generate(self, messages, max_tokens, temperature):
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        payload = {"model": self.model, "max_tokens": max_tokens, "messages": convo, "temperature": temperature}
        if system_text:
            payload["system"] = system_text
        t0 = time.perf_counter()
        r = self._client.post(self.API_URL, json=payload)
        if 400 <= r.status_code < 500 and r.status_code != 429:
            raise LLMError(f"Anthropic request rejected ({r.status_code}): {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return LLMResponse(
            text=text.strip(), model=data.get("model", self.model),
            usage=TokenUsage(prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000, finish_reason=data.get("stop_reason"),
        )


# --------------------------------------------------------------------------------------
# Extractive baseline (no model)
# --------------------------------------------------------------------------------------
_EVIDENCE_BLOCK = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.S | re.M)
_QUESTION = re.compile(r"Question:\s*(.+?)(?:\n\n|\Z)", re.S)


class ExtractiveLLM(LLMClient):
    """Deterministic, model-free 'generator' that quotes the most relevant sentences.

    It reads the numbered evidence blocks produced by `atlas.rag.prompts` and
    returns the top sentences by query-term overlap, each with its citation
    marker. Purpose: (1) a floor baseline showing what retrieval alone achieves,
    (2) fully offline tests, (3) a cheap terminal action inside RL rollouts.
    It cannot follow instructions (planning/critique prompts return "{}"), so the
    agent uses heuristic planner/critic nodes when this provider is selected.
    """

    provider = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        super().__init__("extractive", 0)
        self.max_sentences = max_sentences

    def _generate(self, messages, max_tokens, temperature):
        prompt = messages[-1].content
        qm = _QUESTION.search(prompt)
        blocks = _EVIDENCE_BLOCK.findall(prompt)
        if not qm or not blocks:
            return LLMResponse(text="{}", model=self.model)
        q_terms = set(tokenize(qm.group(1)))
        scored: list[tuple[float, int, str]] = []
        for marker, body in blocks:
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                terms = set(tokenize(sent))
                if len(sent) < 20 or not terms:
                    continue
                overlap = len(q_terms & terms) / (len(q_terms) or 1)
                scored.append((overlap, int(marker), sent.strip()))
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen = [s for s in scored if s[0] > 0][: self.max_sentences]
        if not chosen:
            text = "The provided evidence does not contain enough information to answer this question."
        else:
            text = " ".join(f"{s} [{m}]" for _, m, s in chosen)
        n_prompt = len(prompt.split())
        return LLMResponse(text=text, model=self.model,
                           usage=TokenUsage(prompt_tokens=n_prompt, completion_tokens=len(text.split())))


def build_llm(cfg: LLMConfig, cache: Cache | None = None) -> LLMClient:
    if cfg.provider == "hf_local":
        return HFLocalLLM(cfg, cache)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleLLM(cfg, cache)
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, cache)
    return ExtractiveLLM()

