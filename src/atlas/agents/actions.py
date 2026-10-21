"""The agent's action space, shared by every policy (heuristic, LLM, RL-learned).

    SEARCH_DOCUMENTS  hybrid retrieval for the current sub-question
    LOOKUP_TABLE      structured table rows (SQLite) relevant to the sub-question
    ANALYZE_VISUAL    VLM reading of the most relevant figure
    RUN_PYTHON        exact arithmetic over collected numbers (LLM writes the code)
    SEARCH_WEB        external background search
    REPLAN            reformulate / decompose the unresolved part of the question
    ANSWER            stop researching and write the report

Keeping one discrete action space makes policies directly comparable in the
evaluation harness and makes the RL environment's action space identical to
what the production agent executes.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from atlas.common.types import Modality
from atlas.retrieval.bm25 import tokenize


class Action(StrEnum):
    SEARCH_DOCUMENTS = "search_documents"
    LOOKUP_TABLE = "lookup_table"
    ANALYZE_VISUAL = "analyze_visual"
    RUN_PYTHON = "run_python"
    SEARCH_WEB = "search_web"
    REPLAN = "replan"
    ANSWER = "answer"


ACTIONS: list[Action] = list(Action)
ACTION_TOOL = {
    Action.SEARCH_DOCUMENTS: "search_documents",
    Action.LOOKUP_TABLE: "lookup_table",
    Action.ANALYZE_VISUAL: "analyze_image",
    Action.RUN_PYTHON: "run_python",
    Action.SEARCH_WEB: "search_web",
}


class Decision(BaseModel):
    action: Action
    args: dict[str, Any] = Field(default_factory=dict)  # explicit tool args (LLM policy) — optional
    rationale: str = ""
    policy: str = ""
    fallback: bool = False  # True when an LLM decision was invalid and a fallback was used
    features: list[float] | None = None
    log_prob: float | None = None
    llm_call: dict[str, Any] | None = None  # usage record when an LLM made the decision


# --- lexical cues used by features and heuristics ---------------------------------
TABLE_CUES = re.compile(r"\b(table|score|bleu|accuracy|f1|perplexity|ppl|how many|number of|params?|parameters|"
                        r"result|results|glue|squad|em\b|flops|cost|value|hyperparameters?)\b", re.I)
FIGURE_CUES = re.compile(r"\b(figure|fig\.|chart|plot|graph|curve|diagram|visuali[sz]|shown|trend|axis|image)\b",
                         re.I)
COMPUTE_CUES = re.compile(r"\b(difference|how much (higher|lower|more|less|better|worse)|ratio|percent|"
                          r"percentage|times (more|larger|smaller)|sum|average|mean|compute|calculate|"
                          r"increase|decrease|improve(ment)?\s+(of|by))\b", re.I)
COMPARE_CUES = re.compile(r"\b(compare|comparison|versus|vs\.?|differ|both|and how)\b", re.I)
_NUMBER = re.compile(r"\d+(\.\d+)?")


def query_cues(text: str) -> dict[str, bool]:
    return {
        "table": bool(TABLE_CUES.search(text)),
        "figure": bool(FIGURE_CUES.search(text)),
        "compute": bool(COMPUTE_CUES.search(text)),
        "compare": bool(COMPARE_CUES.search(text)),
    }


def has_numbers(text: str) -> bool:
    return len(_NUMBER.findall(text)) >= 2


def term_coverage(question: str, texts: list[str]) -> float:
    """Fraction of the question's content terms appearing in the given texts."""
    q = set(tokenize(question))
    if not q:
        return 1.0
    pool: set[str] = set()
    for t in texts:
        pool |= set(tokenize(t))
    return len(q & pool) / len(q)


def modality_counts(evidence: list) -> dict[str, int]:
    counts = {m.value: 0 for m in Modality}
    for e in evidence:
        counts[str(e.modality)] = counts.get(str(e.modality), 0) + 1
    return counts
