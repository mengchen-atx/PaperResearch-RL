"""Asynchronous research jobs: job store + queue.

Why async jobs? A research run takes tens of seconds to minutes (multiple LLM
and tool calls). Holding an HTTP request open that long ties up API workers and
breaks behind load balancers with idle timeouts. Instead: POST /research returns
202 + job_id; clients poll GET /research/{id} or stream GET /research/{id}/events.

Two backends behind one interface:
  * memory — asyncio queue + in-process workers. Single-node/dev mode, no deps.
  * redis  — Redis hashes/lists for job state + a work queue consumed by separate
             `atlas worker` processes. API replicas and workers scale
             independently and share state (distributed mode).
Idempotency: an `Idempotency-Key` header maps to an existing job instead of
creating a duplicate (safe client retries).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Protocol

from atlas.common.ids import new_id
from atlas.serving.schemas import JobOut

QUEUE_KEY = "atlas:jobs:queue"
JOB_TTL_S = 7 * 24 * 3600


class JobStore(Protocol):
    def create(self, query: str, policy: str | None, idempotency_key: str | None = None) -> tuple[JobOut, bool]: ...
    def get(self, job_id: str) -> JobOut | None: ...
    def update(self, job_id: str, **fields: Any) -> JobOut: ...
    def append_event(self, job_id: str, event: dict[str, Any]) -> None: ...
    def events(self, job_id: str, start: int = 0) -> list[dict[str, Any]]: ...
    def enqueue(self, job_id: str) -> None: ...
    def dequeue(self, timeout_s: float = 1.0) -> str | None: ...


class MemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobOut] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._idem: dict[str, str] = {}
        self._queue: list[str] = []
        self._cv = threading.Condition()

    def create(self, query, policy, idempotency_key=None):
        with self._cv:
            if idempotency_key and idempotency_key in self._idem:
                return self._jobs[self._idem[idempotency_key]], False
            job = JobOut(job_id=new_id("job_"), status="queued", query=query, policy=policy, created_at=time.time())
            self._jobs[job.job_id] = job
            self._events[job.job_id] = []
            if idempotency_key:
                self._idem[idempotency_key] = job.job_id
            return job, True

    def get(self, job_id):
        return self._jobs.get(job_id)

    def update(self, job_id, **fields):
        with self._cv:
            job = self._jobs[job_id].model_copy(update=fields)
            self._jobs[job_id] = job
            self._cv.notify_all()
            return job

    def append_event(self, job_id, event):
        with self._cv:
            self._events[job_id].append(event)
            self._cv.notify_all()

    def events(self, job_id, start=0):
        return list(self._events.get(job_id, [])[start:])

    def enqueue(self, job_id):
        with self._cv:
            self._queue.append(job_id)
            self._cv.notify_all()

    def dequeue(self, timeout_s=1.0):
        with self._cv:
            if not self._queue:
                self._cv.wait(timeout_s)
            return self._queue.pop(0) if self._queue else None


class RedisJobStore:
    """Job state in Redis: atlas:job:<id> (JSON), atlas:job:<id>:events (list), shared work queue."""

    def __init__(self, client: Any) -> None:
        self.r = client

    def _key(self, job_id: str) -> str:
        return f"atlas:job:{job_id}"

    def create(self, query, policy, idempotency_key=None):
        job = JobOut(job_id=new_id("job_"), status="queued", query=query, policy=policy, created_at=time.time())
        if idempotency_key:
            # SET NX makes idempotency atomic across API replicas.
            ok = self.r.set(f"atlas:idem:{idempotency_key}", job.job_id, nx=True, ex=JOB_TTL_S)
            if not ok:
                existing = self.r.get(f"atlas:idem:{idempotency_key}")
                existing = existing.decode() if isinstance(existing, bytes) else existing
                found = self.get(existing)
                if found:
                    return found, False
        self.r.set(self._key(job.job_id), job.model_dump_json(), ex=JOB_TTL_S)
        return job, True

    def get(self, job_id):
        raw = self.r.get(self._key(job_id))
        return JobOut.model_validate_json(raw) if raw else None

    def update(self, job_id, **fields):
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job = job.model_copy(update=fields)
        self.r.set(self._key(job_id), job.model_dump_json(), ex=JOB_TTL_S)
        return job

    def append_event(self, job_id, event):
        key = f"{self._key(job_id)}:events"
        self.r.rpush(key, json.dumps(event, default=str))
        self.r.expire(key, JOB_TTL_S)

    def events(self, job_id, start=0):
        return [json.loads(x) for x in self.r.lrange(f"{self._key(job_id)}:events", start, -1)]

    def enqueue(self, job_id):
        self.r.lpush(QUEUE_KEY, job_id)

    def dequeue(self, timeout_s=1.0):
        item = self.r.brpop(QUEUE_KEY, timeout=max(1, int(timeout_s)))
        if not item:
            return None
        value = item[1]
        return value.decode() if isinstance(value, bytes) else value


def summarize_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe progress event for SSE clients."""
    ev: dict[str, Any] = {"node": node, "ts": time.time()}
    if node == "planner":
        ev["subquestions"] = update.get("subquestions", [])
    elif node == "router" and update.get("next_action"):
        a = update["next_action"]
        ev.update(action=a["action"], rationale=a.get("rationale", ""), fallback=a.get("fallback", False))
    elif node == "act":
        ev["tool_calls"] = [{"tool": r.tool, "arguments": r.arguments, "ok": r.ok, "error": r.error,
                             "latency_ms": round(r.latency_ms, 1), "new_evidence": r.new_evidence}
                            for r in update.get("tool_history", [])]
        ev["evidence"] = [{"label": e.citation_label(), "modality": str(e.modality), "preview": e.content[:200]}
                          for e in update.get("evidence", [])][:10]
    elif node == "critic":
        ev["critiques"] = update.get("critiques", [])
    elif node == "writer":
        ev["answer"] = update.get("answer", "")
    return ev


def run_research_job(runtime, store: JobStore, job_id: str, max_attempts: int = 2) -> JobOut:
    """Execute one job (called by in-process workers and by `atlas worker`)."""
    job = store.get(job_id)
    if job is None or job.status in ("succeeded", "failed"):
        return job  # already processed (at-least-once delivery -> idempotent handling)
    job = store.update(job_id, status="running", started_at=time.time(), attempts=job.attempts + 1)
    try:
        agent = runtime.agent(job.policy)
        result = None
        for node, update in agent.stream(job.query, session_id=job_id):
            if node == "final":
                result = update["result"]
            else:
                store.append_event(job_id, summarize_event(node, update))
        assert result is not None
        if runtime.cfg.trajectory.enabled:
            runtime.trajectories.append(result.to_trajectory())
        runtime.history.record(result.session_id, result.query, result.answer,
                               [c.model_dump() for c in result.citations], len(result.tool_history), result.status)
        payload = json.loads(result.model_dump_json(exclude={"evidence"}))
        payload["evidence_count"] = len(result.evidence)
        store.append_event(job_id, {"node": "done", "status": result.status, "ts": time.time()})
        return store.update(job_id, status="succeeded", finished_at=time.time(), result=payload)
    except Exception as exc:
        retry = job.attempts < max_attempts
        store.append_event(job_id, {"node": "error", "error": f"{type(exc).__name__}: {exc}", "retry": retry})
        if retry:
            store.update(job_id, status="queued", error=str(exc))
            store.enqueue(job_id)
            return store.get(job_id)
        return store.update(job_id, status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")
