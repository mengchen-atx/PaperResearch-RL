"""Context variables for trace propagation (async- and thread-safe)."""

from __future__ import annotations

from contextvars import ContextVar

_trace_id: ContextVar[str | None] = ContextVar("atlas_trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("atlas_span_id", default=None)


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_span_id() -> str | None:
    return _span_id.get()


def set_trace_id(trace_id: str | None):
    return _trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    _trace_id.reset(token)


def set_span_id(span_id: str | None):
    return _span_id.set(span_id)


def reset_span_id(token) -> None:
    _span_id.reset(token)
