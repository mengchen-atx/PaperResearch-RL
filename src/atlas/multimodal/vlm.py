"""Vision-language model abstraction.

A VLM is only invoked when a question genuinely needs pixels: reading values or
trends off a chart, interpreting a diagram. Captions and tables are handled with
native text extraction, which is cheaper and more accurate.

Implementations:
  * HFLocalVLM       — small local VLM (default SmolVLM-256M) for offline dev.
                        Expect limited chart-reading accuracy at this size.
  * OpenAICompatibleVLM — vLLM-served Qwen2-VL / Llama-3.2-Vision, or OpenAI.
  * AnthropicVLM     — Claude vision via the Messages API.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel

from atlas.common.config import VLMConfig, read_secret
from atlas.common.device import inference_lock, resolve_device
from atlas.common.errors import ConfigError, ToolError
from atlas.common.types import TokenUsage
from atlas.observability.logging import get_logger
from atlas.observability.tracing import get_tracer

log = get_logger("vlm")


class VLMResponse(BaseModel):
    text: str
    model: str
    latency_ms: float
    usage: TokenUsage = TokenUsage()


class VisionLanguageModel(Protocol):
    model: str

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse: ...


def _b64(image_path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return mime, base64.b64encode(image_path.read_bytes()).decode()


class HFLocalVLM:
    def __init__(self, cfg: VLMConfig) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = cfg.model
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        log.info("loading local VLM", extra={"model": cfg.model, "device": self.device})
        self._processor = AutoProcessor.from_pretrained(cfg.model)
        self._model = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=dtype).to(self.device).eval()
        self._lock = inference_lock()

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text_prompt, images=[image], return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        text = self._processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
        return VLMResponse(text=text, model=self.model, latency_ms=(time.perf_counter() - t0) * 1000,
                           usage=TokenUsage(prompt_tokens=int(n_in), completion_tokens=int(out.shape[1] - n_in)))


class OpenAICompatibleVLM:
    def __init__(self, cfg: VLMConfig, transport: httpx.BaseTransport | None = None) -> None:
        if not cfg.base_url:
            raise ConfigError("vlm.base_url is required for provider 'openai_compatible'")
        self.model = cfg.model
        self.cfg = cfg
        headers = {}
        key = read_secret(cfg.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout_s,
                                    transport=transport)

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        mime, data = _b64(image_path)
        payload = {
            "model": self.model, "max_tokens": self.cfg.max_new_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": prompt}]}],
        }
        t0 = time.perf_counter()
        r = self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        body = r.json()
        u = body.get("usage") or {}
        return VLMResponse(text=body["choices"][0]["message"]["content"].strip(), model=self.model,
                           latency_ms=(time.perf_counter() - t0) * 1000,
                           usage=TokenUsage(prompt_tokens=u.get("prompt_tokens", 0),
                                            completion_tokens=u.get("completion_tokens", 0)))


class AnthropicVLM:
    def __init__(self, cfg: VLMConfig, transport: httpx.BaseTransport | None = None) -> None:
        key = read_secret(cfg.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("Set ANTHROPIC_API_KEY to use the anthropic VLM provider")
        self.model = cfg.model
        self.cfg = cfg
        self._client = httpx.Client(
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout=cfg.timeout_s, transport=transport)

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        mime, data = _b64(image_path)
        payload = {"model": self.model, "max_tokens": self.cfg.max_new_tokens, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
            {"type": "text", "text": prompt}]}]}
        t0 = time.perf_counter()
        r = self._client.post("https://api.anthropic.com/v1/messages", json=payload)
        r.raise_for_status()
        body = r.json()
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        u = body.get("usage", {})
        return VLMResponse(text=text.strip(), model=self.model, latency_ms=(time.perf_counter() - t0) * 1000,
                           usage=TokenUsage(prompt_tokens=u.get("input_tokens", 0),
                                            completion_tokens=u.get("output_tokens", 0)))


class DisabledVLM:
    model = "none"

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        raise ToolError("No vision-language model configured (vlm.provider=none)")


class TracedVLM:
    def __init__(self, inner: VisionLanguageModel) -> None:
        self.inner = inner
        self.model = inner.model

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        with get_tracer().span("vlm.analyze", kind="llm", model=self.model, image=image_path.name) as attrs:
            resp = self.inner.analyze(image_path, prompt)
            attrs["latency_ms"] = round(resp.latency_ms, 1)
        return resp


def build_vlm(cfg: VLMConfig) -> VisionLanguageModel:
    if cfg.provider == "none":
        return DisabledVLM()
    if cfg.provider == "hf_local":
        return TracedVLM(HFLocalVLM(cfg))
    if cfg.provider == "openai_compatible":
        return TracedVLM(OpenAICompatibleVLM(cfg))
    return TracedVLM(AnthropicVLM(cfg))
