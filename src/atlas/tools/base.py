"""Tool interface + executor.

Why separate tools from the agent? Each tool is an independently testable unit
with a typed argument schema, a structured result, and uniform instrumentation.
The agent (LLM, heuristic, or RL policy) only decides *which* tool to call with
*which* arguments; the executor handles validation, timeouts, errors, metrics,
and produces a `ToolCallRecord` — the raw material for evaluation (tool-use
metrics) and post-training (trajectories).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.common.errors import AtlasError
from atlas.common.types import Evidence
from atlas.observability.logging import get_logger
from atlas.observability.metrics import TOOL_CALLS, TOOL_LATENCY
from atlas.observability.tracing import get_tracer

log = get_logger("tools")


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    tool: str
    ok: bool
    output: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """One step of tool use, as stored in trajectories and traces."""

    step: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    latency_ms: float
    timestamp: str
    num_evidence: int = 0
    new_evidence: int = 0
    output_preview: str = ""


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[ToolArgs]]
    cost: ClassVar[float] = 1.0  # relative cost unit, used by the RL reward

    def run(self, args: ToolArgs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.Args.model_json_schema()}


class ToolExecutor:
    def __init__(self, tools: list[Tool], timeout_s: float = 60.0) -> None:
        self.tools = {t.name: t for t in tools}
        self.timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="atlas-tool")

    def available(self) -> list[str]:
        return list(self.tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def call(self, name: str, arguments: dict[str, Any], step: int = 0) -> tuple[ToolResult, ToolCallRecord]:
        t0 = time.perf_counter()
        ts = datetime.now(UTC).isoformat()
        with get_tracer().span(f"tool.{name}", kind="tool", tool=name, args=arguments, step=step) as attrs:
            result = self._invoke(name, arguments)
            attrs.update(ok=result.ok, error=result.error, num_evidence=len(result.evidence))
        latency = (time.perf_counter() - t0) * 1000
        result = result.model_copy(update={"latency_ms": latency})
        TOOL_CALLS.labels(name, "ok" if result.ok else "error").inc()
        TOOL_LATENCY.labels(name).observe(latency / 1000)
        record = ToolCallRecord(
            step=step, tool=name, arguments=arguments, ok=result.ok, error=result.error, latency_ms=latency,
            timestamp=ts, num_evidence=len(result.evidence), output_preview=result.output[:300],
        )
        return result, record

    def _invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(tool=name, ok=False, error=f"Unknown tool '{name}'. Available: {self.available()}")
        try:
            args = tool.Args.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(tool=name, ok=False, error=f"Invalid arguments: {exc.errors(include_url=False)}")
        future = self._pool.submit(tool.run, args)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeout:
            return ToolResult(tool=name, ok=False, error=f"Tool timed out after {self.timeout_s:.0f}s")
        except AtlasError as exc:
            return ToolResult(tool=name, ok=False, error=str(exc))
        except Exception as exc:  # a tool bug must not crash the agent; it is recorded instead
            log.exception("tool crashed", extra={"tool": name})
            return ToolResult(tool=name, ok=False, error=f"{type(exc).__name__}: {exc}")
