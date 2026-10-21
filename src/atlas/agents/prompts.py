"""Prompts for the LLM-driven agent nodes. All outputs are JSON, validated with fallbacks."""

from __future__ import annotations

from atlas.common.types import Evidence
from atlas.llm.base import Message, system, user

PLANNER_SYSTEM = (
    "You plan research over a library of scientific papers. Break the question into at most {n} short, "
    "self-contained sub-questions that can each be answered by searching the papers. If the question is "
    "simple, return it as a single sub-question. Respond with JSON only: "
    '{{"subquestions": ["..."], "rationale": "..."}}'
)

ROUTER_SYSTEM = (
    "You are the router of a research agent. Choose the single next action.\n"
    "Actions:\n"
    "- search_documents: search the papers. args: {{\"query\": str}}\n"
    "- lookup_table: fetch a relevant data table. args: {{\"query\": str}}\n"
    "- analyze_visual: read a figure/chart with a vision model. args: {{\"question\": str}}\n"
    "- run_python: exact arithmetic over numbers already in the evidence. args: {{\"code\": str}}\n"
    "- search_web: search the web. args: {{\"query\": str}}\n"
    "- replan: re-decompose the question when research is stuck. args: {{}}\n"
    "- answer: enough evidence has been collected. args: {{}}\n"
    "Only these actions are currently available: {available}.\n"
    'Respond with JSON only: {{"action": "...", "args": {{...}}, "reason": "..."}}'
)

CRITIC_SYSTEM = (
    "You check whether collected evidence is sufficient to answer a research question with citations. "
    "Be strict: numbers and specific claims must appear in the evidence. Respond with JSON only: "
    '{"sufficient": true|false, "missing": "what is still missing, or empty"}'
)

PYTHON_SYSTEM = (
    "Write a short Python program that computes the answer to the question using ONLY numbers that appear "
    "in the evidence. Use only the math/statistics modules, no input or files. print() the result with a "
    "short label. Output only code, no explanation."
)

WRITER_INSTRUCTIONS = (
    "Write a direct answer in 2-5 sentences. Cite every claim with [n]. If computation evidence is "
    "present, use its result. If evidence is missing for part of the question, say what is missing."
)


def _evidence_digest(evidence: list[Evidence], limit: int = 8, chars: int = 220) -> str:
    if not evidence:
        return "(none yet)"
    return "\n".join(f"[{i}] ({e.modality}, {e.citation_label()}) {e.content[:chars]}"
                     for i, e in enumerate(evidence[-limit:], start=1))


def planner_messages(query: str, n: int, missing: str = "") -> list[Message]:
    body = f"Question: {query}"
    if missing:
        body += f"\nPrevious research was insufficient. Still missing: {missing}\nPlan sub-questions for the missing parts."
    return [system(PLANNER_SYSTEM.format(n=n)), user(body)]


def router_messages(query: str, subquestion: str, evidence: list[Evidence], history: list[str],
                    available: list[str]) -> list[Message]:
    body = (f"Research question: {query}\nCurrent sub-question: {subquestion}\n\n"
            f"Evidence so far:\n{_evidence_digest(evidence)}\n\n"
            f"Previous actions: {', '.join(history) or 'none'}\n\nNext action JSON:")
    return [system(ROUTER_SYSTEM.format(available=", ".join(available))), user(body)]


def critic_messages(query: str, evidence: list[Evidence]) -> list[Message]:
    return [system(CRITIC_SYSTEM),
            user(f"Question: {query}\n\nEvidence:\n{_evidence_digest(evidence, limit=10, chars=400)}\n\nJSON:")]


def python_messages(question: str, evidence: list[Evidence]) -> list[Message]:
    return [system(PYTHON_SYSTEM),
            user(f"Question: {question}\n\nEvidence:\n{_evidence_digest(evidence, limit=6, chars=900)}\n\nCode:")]
