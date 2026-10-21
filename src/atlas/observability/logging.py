"""Structured logging with request/trace correlation.

Every log record is enriched with the current `trace_id` (from a contextvar set
by the tracer or the API middleware), so a single research request can be
followed across API -> agent node -> tool -> LLM call in the logs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from atlas.observability.context import current_trace_id

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        tid = current_trace_id()
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        if tid:
            base += f" [trace={tid[:8]}]"
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root = logging.getLogger("atlas")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False
    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("httpx", "sentence_transformers", "transformers", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("atlas") else f"atlas.{name}")
