"""Queue worker for distributed mode: `atlas worker` (one process per replica).

Consumes job ids from the Redis work queue and executes them with the same
`run_research_job` used by the API's in-process workers. Delivery is
at-least-once; `run_research_job` skips jobs already finished (idempotent), and
failed attempts are re-queued up to `max_attempts`. SIGTERM (Kubernetes pod
shutdown) stops taking new jobs and lets the current one finish.
"""

from __future__ import annotations

import signal
import threading

from atlas.common.config import AtlasConfig
from atlas.observability.logging import get_logger
from atlas.observability.tracing import configure_tracer

log = get_logger("worker")


def run_worker(cfg: AtlasConfig, max_jobs: int | None = None) -> int:
    from atlas.runtime import AtlasRuntime
    from atlas.serving.api import build_job_store
    from atlas.serving.jobs import run_research_job

    if cfg.serving.queue_backend != "redis":
        log.warning("worker started with queue_backend=memory; it only sees jobs created in this process")
    configure_tracer(cfg.resolve_path(cfg.paths.trace_dir))
    rt = AtlasRuntime(cfg)
    store = build_job_store(cfg)
    stop = threading.Event()

    def _graceful(signum, _frame) -> None:
        log.info("shutdown signal received; finishing current job", extra={"signal": signum})
        stop.set()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    processed = 0
    log.info("worker ready", extra={"backend": cfg.serving.queue_backend})
    while not stop.is_set() and (max_jobs is None or processed < max_jobs):
        job_id = store.dequeue(timeout_s=1.0)
        if not job_id:
            continue
        job = run_research_job(rt, store, job_id)
        processed += 1
        log.info("job processed", extra={"job_id": job_id, "status": job.status if job else None})
    return processed
