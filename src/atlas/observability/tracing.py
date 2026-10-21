"""Minimal, vendor-neutral tracing.

Why not OpenTelemetry directly? OTel is the right choice for a multi-service
production deployment, but it adds an SDK + collector to every local run. This
tracer records the same shape of data (trace_id, span_id, parent, name, timing,
attributes, status) into an in-process buffer and optional JSONL files, which is
enough to inspect an agent trajectory after execution. Spans can be exported to
an OTel collector later by writing an exporter for `SpanRecord`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atlas.observability import context as ctx
from atlas.observability.logging import get_logger

log = get_logger("tracing")


class SpanRecord(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str  # request | agent_node | tool | retrieval | llm | other
    start_ts: float
    duration_ms: float
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Tracer:
    def __init__(self, export_dir: Path | None = None, max_traces: int = 500) -> None:
        self._export_dir = export_dir
        self._max_traces = max_traces
        self._spans: dict[str, list[SpanRecord]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def trace(self, name: str, trace_id: str | None = None, **attrs: Any) -> Iterator[str]:
        """Start a new trace (root span). Yields the trace_id."""
        tid = trace_id or uuid.uuid4().hex
        token = ctx.set_trace_id(tid)
        try:
            with self.span(name, kind="request", **attrs):
                yield tid
        finally:
            ctx.reset_trace_id(token)
            self._export(tid)

    @contextmanager
    def span(self, name: str, kind: str = "other", **attrs: Any) -> Iterator[dict[str, Any]]:
        """Record a child span. Yields a mutable attribute dict callers may enrich."""
        tid = ctx.current_trace_id()
        if tid is None:  # no active trace: tracing is a no-op
            yield {}
            return
        sid = uuid.uuid4().hex[:16]
        parent = ctx.current_span_id()
        token = ctx.set_span_id(sid)
        start = time.time()
        t0 = time.perf_counter()
        attributes: dict[str, Any] = dict(attrs)
        status, error = "ok", None
        try:
            yield attributes
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ctx.reset_span_id(token)
            record = SpanRecord(
                trace_id=tid,
                span_id=sid,
                parent_span_id=parent,
                name=name,
                kind=kind,
                start_ts=start,
                duration_ms=(time.perf_counter() - t0) * 1000,
                status=status,
                error=error,
                attributes={k: _jsonable(v) for k, v in attributes.items()},
            )
            with self._lock:
                self._spans[tid].append(record)
                if len(self._spans) > self._max_traces:
                    oldest = next(iter(self._spans))
                    self._spans.pop(oldest, None)

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        with self._lock:
            spans = list(self._spans.get(trace_id, []))
        if not spans and self._export_dir:
            path = self._export_dir / f"{trace_id}.jsonl"
            if path.exists():
                spans = [SpanRecord.model_validate_json(line) for line in path.read_text().splitlines()]
        return sorted(spans, key=lambda s: s.start_ts)

    def _export(self, trace_id: str) -> None:
        if not self._export_dir:
            return
        try:
            self._export_dir.mkdir(parents=True, exist_ok=True)
            spans = self.get_trace(trace_id)
            with (self._export_dir / f"{trace_id}.jsonl").open("w") as f:
                for s in spans:
                    f.write(s.model_dump_json() + "\n")
        except OSError:
            log.exception("failed to export trace", extra={"trace_id_export": trace_id})


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


_TRACER = Tracer()


def get_tracer() -> Tracer:
    return _TRACER


def configure_tracer(export_dir: Path | None) -> Tracer:
    global _TRACER
    _TRACER = Tracer(export_dir=export_dir)
    return _TRACER
