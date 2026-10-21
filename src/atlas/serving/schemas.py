"""API request/response schemas (validated at the system boundary)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas.common.types import Modality


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveRequest(Strict):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    modality: Modality | None = None
    filename: str | None = Field(None, max_length=300)


class EvidenceOut(BaseModel):
    evidence_id: str
    modality: str
    label: str
    filename: str | None
    page: int | None
    score: float | None
    content: str


class RetrieveResponse(BaseModel):
    query: str
    results: list[EvidenceOut]
    latency_ms: float


class AskRequest(Strict):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]]
    evidence: list[EvidenceOut]
    invalid_citations: list[int]
    latency_ms: float
    usage: dict[str, int]
    model: str


class ResearchRequest(Strict):
    query: str = Field(min_length=3, max_length=2000)
    policy: Literal["heuristic", "llm", "learned"] | None = None
    mode: Literal["sync", "async"] = "async"


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    query: str
    policy: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
