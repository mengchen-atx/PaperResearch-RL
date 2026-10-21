import json

import httpx
import pytest

from atlas.common.cache import MemoryCache
from atlas.common.config import LLMConfig
from atlas.common.errors import ConfigError, LLMError
from atlas.common.types import Evidence, Modality
from atlas.llm.base import Message, user
from atlas.llm.providers import AnthropicLLM, ExtractiveLLM, OpenAICompatibleLLM
from atlas.rag.prompts import grounded_messages


def _openai_handler(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": t}}]} for t in ("Hel", "lo")]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"model": body["model"], "choices": [
            {"message": {"content": "  pong  "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    return handler


def test_openai_compatible_generate_stream_and_cache():
    calls = []
    cfg = LLMConfig(provider="openai_compatible", model="m", base_url="http://x/v1")
    llm = OpenAICompatibleLLM(cfg, cache=MemoryCache(), transport=httpx.MockTransport(_openai_handler(calls)))
    r = llm.generate([user("ping")])
    assert r.text == "pong" and r.usage.prompt_tokens == 5
    assert llm.generate([user("ping")]).cached and len(calls) == 1  # temperature 0 -> cached
    assert "".join(llm.stream([user("hi")])) == "Hello"


def test_openai_compatible_requires_base_url():
    with pytest.raises(ConfigError):
        OpenAICompatibleLLM(LLMConfig(provider="openai_compatible", model="m"))


def test_retries_on_5xx_then_succeeds_and_4xx_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    cfg = LLMConfig(provider="openai_compatible", model="m", base_url="http://x/v1", max_retries=2)
    assert OpenAICompatibleLLM(cfg, transport=httpx.MockTransport(flaky)).generate([user("q")]).text == "ok"
    bad = OpenAICompatibleLLM(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")))
    with pytest.raises(LLMError, match="rejected"):
        bad.generate([user("q")])


def test_anthropic_provider_payload(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-api-key"]
        return httpx.Response(200, json={"model": "claude", "content": [{"type": "text", "text": "hi"}],
                                         "usage": {"input_tokens": 3, "output_tokens": 1}, "stop_reason": "end_turn"})

    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="claude-haiku-4-5"), transport=httpx.MockTransport(handler))
    r = llm.generate([Message(role="system", content="sys"), user("hello")])
    assert r.text == "hi" and seen["body"]["system"] == "sys" and seen["key"] == "test-key"
    assert all(m["role"] != "system" for m in seen["body"]["messages"])


def test_extractive_llm_quotes_and_cites_evidence():
    ev = [Evidence(evidence_id="1", modality=Modality.TEXT, content="Cats sleep a lot. The capital of Zedland is Quorra.",
                   source_tool="t", filename="f.txt")]
    r = ExtractiveLLM().generate(grounded_messages("What is the capital of Zedland?", ev))
    assert "Quorra" in r.text and "[1]" in r.text
    assert ExtractiveLLM().generate([user("plan this")]).text == "{}"
