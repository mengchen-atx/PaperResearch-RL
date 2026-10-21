"""Key-value caches for retrieval results and deterministic LLM responses.

Where caching is safe in Atlas:
  * retrieval results for (query, retrieval-config) — the index is immutable per build
  * LLM responses only when temperature == 0 (deterministic decoding)
Where it is NOT used: sampled LLM outputs, tool calls with side effects, web search
(results change over time; a short TTL would be needed).

`MemoryCache` serves single-process/local mode; `RedisCache` shares the cache
across API replicas and workers in the distributed deployment.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

from atlas.common.config import CacheConfig
from atlas.common.ids import sha256_text
from atlas.observability.logging import get_logger
from atlas.observability.metrics import CACHE_EVENTS

log = get_logger("cache")


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None: ...


def make_key(namespace: str, payload: Any) -> str:
    return f"atlas:{namespace}:{sha256_text(json.dumps(payload, sort_keys=True, default=str))[:32]}"


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        return None


class MemoryCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, max_items: int = 2048, ttl_s: int = 3600, name: str = "memory") -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_items
        self._ttl = ttl_s
        self._name = name
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < time.time():
                self._data.pop(key, None)
                CACHE_EVENTS.labels(self._name, "miss").inc()
                return None
            self._data.move_to_end(key)
        CACHE_EVENTS.labels(self._name, "hit").inc()
        return item[1]

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        expires = time.time() + (ttl_s or self._ttl)
        with self._lock:
            self._data[key] = (expires, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


class RedisCache:
    """JSON-serialized cache in Redis. Failures degrade to cache misses, never errors."""

    def __init__(self, client: Any, ttl_s: int = 3600, name: str = "redis") -> None:
        self._r = client
        self._ttl = ttl_s
        self._name = name

    def get(self, key: str) -> Any | None:
        try:
            raw = self._r.get(key)
        except Exception:  # redis down -> behave like a miss
            log.warning("redis cache get failed", exc_info=True)
            return None
        CACHE_EVENTS.labels(self._name, "hit" if raw is not None else "miss").inc()
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        try:
            self._r.set(key, json.dumps(value, default=str), ex=ttl_s or self._ttl)
        except Exception:
            log.warning("redis cache set failed", exc_info=True)


def build_cache(cfg: CacheConfig, redis_url: str | None = None) -> Cache:
    if cfg.backend == "none":
        return NullCache()
    if cfg.backend == "redis":
        import redis

        return RedisCache(redis.Redis.from_url(redis_url or "redis://localhost:6379/0"), ttl_s=cfg.ttl_s)
    return MemoryCache(max_items=cfg.max_items, ttl_s=cfg.ttl_s)
