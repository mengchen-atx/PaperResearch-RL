"""Trajectory -> SFT examples.

We post-train the *agent's* LLM on its own successful behavior (rejection
sampling / expert iteration style): only trajectories that answered correctly
with valid citations pass `quality_filter`. Two example types per trajectory:

  router  (state at step t) -> {"action", "args", "reason"} JSON
          The state is rebuilt exactly: evidence before step t is the prefix of
          the trajectory's evidence list (evidence is append-only and each step
          records how many new items it added).
  writer  grounded prompt with the writer's evidence -> final cited answer

Examples use TRL's conversational prompt/completion format so the loss is
computed on the completion (assistant) tokens only.
"""

from __future__ import annotations

import json
import random
from typing import Any

from atlas.agents.actions import ACTIONS, Action
from atlas.agents.prompts import WRITER_INSTRUCTIONS, router_messages
from atlas.rag.prompts import grounded_messages
from atlas.trajectory.schema import Trajectory

# In the training data every action is listed as available; at inference the
# prompt lists the currently unmasked ones (same template).
_ALL_ACTIONS = [a.value for a in ACTIONS]


def _msgs(messages) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def router_examples(t: Trajectory) -> list[dict[str, Any]]:
    examples = []
    n_evidence = 0
    history: list[str] = []
    for step in t.steps:
        prompt = _msgs(router_messages(t.query, step.subquestion or t.query, t.evidence[:n_evidence], history,
                                       _ALL_ACTIONS))
        args = {k: v for k, v in step.arguments.items() if k in ("query", "question", "code")}
        target = {"action": step.action, "args": args if step.action != Action.ANSWER else {},
                  "reason": step.rationale[:200]}
        examples.append({"prompt": prompt, "completion": [{"role": "assistant", "content": json.dumps(target)}],
                         "kind": "router", "trajectory_id": t.trajectory_id, "gold_action": step.action})
        history.append(step.action)
        if step.observation is not None:
            n_evidence += step.observation.new_evidence
    return examples


def writer_example(t: Trajectory) -> dict[str, Any] | None:
    ids = t.metadata.get("writer_evidence_ids") or []
    by_id = {e.evidence_id: e for e in t.evidence}
    evidence = [by_id[i] for i in ids if i in by_id]
    if not evidence or not t.final_answer:
        return None
    prompt = _msgs(grounded_messages(t.query, evidence, WRITER_INSTRUCTIONS))
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": t.final_answer}], "kind": "writer",
            "trajectory_id": t.trajectory_id, "gold_action": None}


def trajectories_to_examples(trajs: list[Trajectory], include_writer: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in trajs:
        out.extend(router_examples(t))
        if include_writer:
            w = writer_example(t)
            if w:
                out.append(w)
    return out


def split_by_trajectory(examples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by trajectory id so steps of one episode never straddle train/val (no leakage)."""
    ids = sorted({e["trajectory_id"] for e in examples})
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    return ([e for e in examples if e["trajectory_id"] not in val_ids],
            [e for e in examples if e["trajectory_id"] in val_ids])
