"""Citation parsing and validation.

A citation `[n]` is *valid* when n refers to an evidence item that was actually
shown to the model. Markers outside 1..len(evidence) are hallucinated citations
and are reported (and stripped from the resolved list) — Atlas never invents a
source that retrieval did not return.
"""

from __future__ import annotations

import re

from atlas.common.types import Citation, Evidence

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_markers(text: str) -> list[int]:
    """All citation numbers in order of first appearance. Handles [1], [1, 2], [1][2]."""
    seen: list[int] = []
    for group in _MARKER.findall(text):
        for n in group.split(","):
            v = int(n.strip())
            if v not in seen:
                seen.append(v)
    return seen


def resolve_citations(answer: str, evidence: list[Evidence]) -> tuple[list[Citation], list[int]]:
    """Map markers to evidence. Returns (valid citations, invalid marker numbers)."""
    valid: list[Citation] = []
    invalid: list[int] = []
    for n in extract_markers(answer):
        if 1 <= n <= len(evidence):
            ev = evidence[n - 1]
            valid.append(Citation(marker=n, evidence_id=ev.evidence_id, label=ev.citation_label(),
                                  quote=ev.content[:240]))
        else:
            invalid.append(n)
    return valid, invalid


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims for citation-coverage metrics."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer.strip())
    return [p for p in (s.strip() for s in parts) if len(p) > 3]
