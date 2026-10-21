"""Agent trajectory schema — the bridge between the agent, evaluation, SFT and RL.

A trajectory records one research episode as a sequence of (state, action,
observation) steps plus the final answer and (optionally) its evaluation:

    query -> plan -> [step_t = (state summary, action, tool, args, observation, latency)]* -> answer

JSONL (one trajectory per line) is used because it is append-only (safe for
concurrent writers with a lock), streamable, diff-able, and is the native input
format of HF `datasets` for SFT.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from atlas.common.types import Citation, Evidence

SCHEMA_VERSION = "1.0"


class StepObservation(BaseModel):
    ok: bool = True
    output: str = ""
    error: str | None = None
    new_evidence: int = 0


class TrajectoryStep(BaseModel):
    index: int
    subquestion: str = ""
    action: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: StepObservation | None = None
    policy: str = ""
    rationale: str = ""
    fallback: bool = False
    features: list[float] | None = None
    log_prob: float | None = None
    latency_ms: float = 0.0
    reward: float | None = None


class Trajectory(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trajectory_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    query: str
    session_id: str = ""
    plan: list[str] = Field(default_factory=list)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    final_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    status: str = ""
    policy: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    evaluation: dict[str, Any] | None = None
    success: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
