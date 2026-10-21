"""Prometheus metrics. Exposed at /metrics by the API; scraped by Prometheus in prod."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)

REQUESTS = Counter(
    "atlas_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "atlas_request_latency_seconds", "API request latency", ["endpoint"],
    buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
TOOL_CALLS = Counter(
    "atlas_tool_calls_total", "Agent tool calls", ["tool", "status"], registry=REGISTRY
)
TOOL_LATENCY = Histogram(
    "atlas_tool_latency_seconds", "Tool latency", ["tool"], buckets=_LATENCY_BUCKETS, registry=REGISTRY
)
LLM_TOKENS = Counter(
    "atlas_llm_tokens_total", "LLM tokens", ["model", "kind"], registry=REGISTRY
)
LLM_LATENCY = Histogram(
    "atlas_llm_latency_seconds", "LLM call latency", ["model"], buckets=_LATENCY_BUCKETS, registry=REGISTRY
)
RETRIEVAL_LATENCY = Histogram(
    "atlas_retrieval_latency_seconds", "Retrieval latency", ["mode"],
    buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
CACHE_EVENTS = Counter(
    "atlas_cache_events_total", "Cache hits/misses", ["cache", "event"], registry=REGISTRY
)
JOBS = Counter("atlas_jobs_total", "Async research jobs", ["status"], registry=REGISTRY)
