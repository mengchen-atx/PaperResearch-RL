"""FastAPI service.

Endpoints
    GET  /health                     liveness (process up)
    GET  /ready                      readiness (index loaded)
    POST /retrieve                   hybrid retrieval only
    POST /ask                        single-shot RAG with citations
    POST /research                   research agent (mode=async -> 202 + job; mode=sync -> result)
    GET  /research/{job_id}          job status/result
    GET  /research/{job_id}/events   Server-Sent Events progress stream
    GET  /traces/{trace_id}          spans of one request (agent nodes, tools, LLM calls)
    GET  /sessions                   recent research sessions (persistent history)
    POST /documents                  upload a document (validated) and re-index in the background
    GET  /documents/status           ingestion status
    GET  /metrics                    Prometheus metrics
    GET  /                           minimal web UI

Blocking work (model inference, tools) runs in a thread pool with a timeout so the
event loop stays responsive. Every response carries X-Request-ID, which is also
the trace id for synchronous endpoints.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from atlas.common.config import AtlasConfig
from atlas.common.errors import AtlasError, IndexNotFoundError, UnsafePathError
from atlas.common.ids import new_id
from atlas.common.paths import safe_resolve
from atlas.common.types import Evidence
from atlas.ingestion.loaders import SUPPORTED_SUFFIXES
from atlas.observability import context as octx
from atlas.observability.logging import configure_logging, get_logger
from atlas.observability.metrics import JOBS, REGISTRY, REQUEST_LATENCY, REQUESTS
from atlas.observability.tracing import configure_tracer, get_tracer
from atlas.retrieval.filters import MetadataFilter
from atlas.serving.jobs import JobStore, MemoryJobStore, RedisJobStore, run_research_job
from atlas.serving.schemas import (
    AskRequest,
    AskResponse,
    EvidenceOut,
    JobOut,
    ResearchRequest,
    RetrieveRequest,
    RetrieveResponse,
)

log = get_logger("api")
UI_PATH = Path(__file__).parent / "static" / "index.html"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _evidence_out(e: Evidence) -> EvidenceOut:
    return EvidenceOut(evidence_id=e.evidence_id, modality=str(e.modality), label=e.citation_label(),
                       filename=e.filename, page=e.page, score=e.score, content=e.content[:2000])


class RateLimiter:
    """Sliding-window per-client limiter (in-process). Behind multiple replicas, enforce
    limits at the gateway (see deployment/kubernetes/ingress.yaml) or use a Redis counter."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def build_job_store(cfg: AtlasConfig) -> JobStore:
    if cfg.serving.queue_backend == "redis":
        import redis

        return RedisJobStore(redis.Redis.from_url(cfg.serving.redis_url))
    return MemoryJobStore()


def create_app(cfg: AtlasConfig, runtime=None, job_store: JobStore | None = None) -> FastAPI:
    from atlas.runtime import AtlasRuntime

    configure_logging(cfg.observability.log_level, cfg.observability.log_json)
    configure_tracer(cfg.resolve_path(cfg.paths.trace_dir) if cfg.observability.tracing else None)
    rt = runtime or AtlasRuntime(cfg)
    store = job_store or build_job_store(cfg)
    limiter = RateLimiter(cfg.serving.rate_limit_per_minute)
    stop = threading.Event()
    ingest_status: dict[str, Any] = {"state": "idle"}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if cfg.serving.queue_backend == "memory":
            for i in range(cfg.serving.max_concurrent_jobs):
                threading.Thread(target=worker_loop, name=f"atlas-job-worker-{i}", daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="Atlas Research API", version="0.1.0", lifespan=lifespan)
    app.state.runtime, app.state.store = rt, store

    # --- in-process workers (memory backend only; redis mode uses `atlas worker`) ---------
    def worker_loop() -> None:
        while not stop.is_set():
            job_id = store.dequeue(timeout_s=0.5)
            if job_id:
                job = run_research_job(rt, store, job_id)
                JOBS.labels(job.status if job else "missing").inc()

    # --- middleware: request id, rate limit, metrics ---------------------------------------
    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_id()
        token = octx.set_trace_id(rid)
        endpoint = request.url.path.split("/")[1] or "root"
        t0 = time.perf_counter()
        try:
            client = request.client.host if request.client else "unknown"
            if endpoint not in ("health", "ready", "metrics") and not limiter.allow(client):
                response = JSONResponse({"error": "rate limit exceeded", "request_id": rid}, status_code=429)
            else:
                response = await call_next(request)
        finally:
            octx.reset_trace_id(token)
        response.headers["X-Request-ID"] = rid
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
        return response

    @app.exception_handler(AtlasError)
    async def _atlas_error(request: Request, exc: AtlasError):
        status = 404 if isinstance(exc, IndexNotFoundError) else 400
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc),
                             "request_id": octx.current_trace_id()}, status_code=status)

    async def _run(fn, *args, timeout: float | None = None):
        try:
            return await asyncio.wait_for(run_in_threadpool(fn, *args), timeout or cfg.serving.request_timeout_s)
        except TimeoutError as exc:
            raise HTTPException(504, "request timed out") from exc

    # --- health -------------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        try:
            n = len(rt.index.chunks)
        except IndexNotFoundError as exc:
            return JSONResponse({"status": "not_ready", "detail": str(exc)}, status_code=503)
        return JSONResponse({"status": "ready", "chunks": n, "queue_backend": cfg.serving.queue_backend})

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # --- retrieval & RAG ---------------------------------------------------------------------
    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        flt = MetadataFilter(modalities=[req.modality] if req.modality else None,
                             filenames=[req.filename] if req.filename else None)
        t0 = time.perf_counter()
        hits = await _run(rt.retriever.retrieve, req.query, req.top_k, flt)
        return RetrieveResponse(query=req.query, latency_ms=(time.perf_counter() - t0) * 1000,
                                results=[_evidence_out(Evidence.from_scored_chunk(h, "retrieve")) for h in hits])

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        res = await _run(rt.rag().answer, req.question, req.top_k)
        return AskResponse(question=res.question, answer=res.answer, citations=[c.model_dump() for c in res.citations],
                           evidence=[_evidence_out(e) for e in res.evidence], invalid_citations=res.invalid_citations,
                           latency_ms=res.total_ms, model=res.model,
                           usage={"prompt_tokens": res.usage.prompt_tokens,
                                  "completion_tokens": res.usage.completion_tokens})

    # --- research agent ------------------------------------------------------------------
    @app.post("/research", status_code=202)
    async def research(req: ResearchRequest, idempotency_key: str | None = Header(None)) -> JSONResponse:
        if req.mode == "sync":
            result = await _run(lambda: rt.agent(req.policy).run(req.query), timeout=cfg.agent.timeout_s + 30)
            return JSONResponse(json.loads(result.model_dump_json()), status_code=200)
        job, created = store.create(req.query, req.policy, idempotency_key)
        if created:
            store.enqueue(job.job_id)
            JOBS.labels("queued").inc()
        return JSONResponse({"job_id": job.job_id, "status": job.status, "created": created,
                             "status_url": f"/research/{job.job_id}", "events_url": f"/research/{job.job_id}/events"},
                            status_code=202)

    @app.get("/research/{job_id}", response_model=JobOut)
    def job_status(job_id: str) -> JobOut:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return job

    @app.get("/research/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if store.get(job_id) is None:
            raise HTTPException(404, f"job {job_id} not found")

        async def gen():
            sent, deadline = 0, time.monotonic() + cfg.agent.timeout_s + 60
            while time.monotonic() < deadline:
                events = await run_in_threadpool(store.events, job_id, sent)
                for ev in events:
                    yield f"event: {ev.get('node', 'message')}\ndata: {json.dumps(ev, default=str)}\n\n"
                sent += len(events)
                job = store.get(job_id)
                if job and job.status in ("succeeded", "failed") and not events:
                    yield f"event: end\ndata: {json.dumps({'status': job.status})}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/traces/{trace_id}")
    def trace(trace_id: str) -> list[dict[str, Any]]:
        spans = get_tracer().get_trace(trace_id)
        if not spans:
            raise HTTPException(404, "trace not found")
        return [s.model_dump() for s in spans]

    @app.get("/sessions")
    def sessions(limit: int = 20) -> list[dict[str, Any]]:
        return rt.history.recent(min(max(limit, 1), 200))

    # --- documents -----------------------------------------------------------------------
    upload_dir = cfg.resolve_path(cfg.paths.upload_dir)

    def _reindex() -> None:
        ingest_status.update(state="running", started_at=time.time())
        try:
            report = rt.ingest([cfg.resolve_path(cfg.paths.data_dir) / "raw"])
            ingest_status.update(state="done", summary=report.summary(), finished_at=time.time())
        except Exception as exc:
            log.exception("re-index failed")
            ingest_status.update(state="failed", error=str(exc), finished_at=time.time())

    @app.post("/documents", status_code=202)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _SAFE_NAME.sub("_", Path(file.filename or "upload").name)[:120]
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES | {".png", ".jpg", ".jpeg"}:
            raise HTTPException(415, f"unsupported file type; allowed: {sorted(SUPPORTED_SUFFIXES)} + images")
        data = await file.read(cfg.serving.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > cfg.serving.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"file exceeds {cfg.serving.max_upload_mb} MB")
        try:
            target = safe_resolve(name, upload_dir)
        except UnsafePathError as exc:
            raise HTTPException(400, str(exc)) from exc
        upload_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if target.suffix.lower() in SUPPORTED_SUFFIXES and ingest_status.get("state") != "running":
            threading.Thread(target=_reindex, daemon=True).start()
        return {"saved": name, "bytes": len(data), "reindex": ingest_status.get("state")}

    @app.get("/documents/status")
    def documents_status() -> dict[str, Any]:
        return ingest_status

    @app.get("/", response_class=HTMLResponse)
    def ui() -> str:
        return UI_PATH.read_text() if UI_PATH.exists() else "<h1>Atlas API</h1><p>See /docs</p>"

    return app
