"""Prompt construction for grounded generation.

The context is a numbered list of evidence blocks:

    [1] (arxiv_1706.03762.pdf, p.5)
    <chunk text>

The model is instructed to cite with the bracket numbers. Numbered markers are
short (cheap in tokens), unambiguous to parse, and map 1:1 back to evidence
objects, which makes citation accuracy *measurable* (see atlas.rag.citations).
"""

from __future__ import annotations

import re

from atlas.common.types import Evidence
from atlas.llm.base import Message, system, user

# Source papers contain their own bibliography markers ("[34, 2, 9]") that collide
# with our [n] citation syntax; small models copy them verbatim. Rewrite them to
# "(ref 34, 2, 9)" inside the context so every [n] the model emits is ours.
_SOURCE_REF = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def neutralize_source_refs(text: str) -> str:
    return _SOURCE_REF.sub(lambda m: f"(ref {m.group(1)})", text)

MAX_EVIDENCE_CHARS = 1400

GROUNDED_SYSTEM = (
    "You are a careful research assistant. Answer the question using ONLY the numbered evidence. "
    "Cite every factual claim with the evidence number in square brackets, e.g. [1] or [2][3]. "
    "If the evidence is insufficient, say so explicitly. Be concise: 1-4 sentences."
)

CLOSED_BOOK_SYSTEM = (
    "You are a knowledgeable research assistant. Answer the question concisely (1-4 sentences) "
    "from your own knowledge."
)


def format_evidence(evidence: list[Evidence], max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    blocks = []
    for i, ev in enumerate(evidence, start=1):
        content = neutralize_source_refs(ev.content)
        body = content if len(content) <= max_chars else content[:max_chars] + " ..."
        blocks.append(f"[{i}] ({ev.citation_label()})\n{body}")
    return "\n\n".join(blocks)


def grounded_messages(question: str, evidence: list[Evidence], instructions: str = "") -> list[Message]:
    body = f"Evidence:\n{format_evidence(evidence)}\n\nQuestion: {question}\n\n"
    if instructions:
        body += f"{instructions}\n\n"
    body += "Answer (with [n] citations):"
    return [system(GROUNDED_SYSTEM), user(body)]


def closed_book_messages(question: str) -> list[Message]:
    return [system(CLOSED_BOOK_SYSTEM), user(f"Question: {question}\n\nAnswer:")]


QUERY_REWRITE_SYSTEM = (
    "Rewrite the user's question into a concise search query for a scientific paper search engine. "
    "Keep technical terms, model names and numbers. Output only the query."
)


def rewrite_messages(question: str) -> list[Message]:
    return [system(QUERY_REWRITE_SYSTEM), user(question)]
