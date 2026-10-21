"""Evaluate a (base or LoRA-adapted) model on held-out router examples.

Metrics (on validation trajectories never seen in training):
    json_valid_rate   output parses as a JSON object with an "action" field
    action_accuracy   predicted action == action taken in the successful trajectory
This directly measures whether SFT taught the small model the agent's tool-use
protocol. End-to-end task success of the SFT agent is measured separately with
configs/experiments/agent_sft.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.common.config import LLMConfig
from atlas.llm.base import Message
from atlas.llm.json_utils import parse_json_object
from atlas.llm.providers import HFLocalLLM


def evaluate_router_imitation(base_model: str, adapter_path: str | None, examples: list[dict[str, Any]],
                              max_examples: int | None = None) -> dict[str, Any]:
    router = [e for e in examples if e["kind"] == "router"][:max_examples]
    llm = HFLocalLLM(LLMConfig(provider="hf_local", model=base_model, adapter_path=adapter_path))
    valid = correct = 0
    confusion: dict[str, dict[str, int]] = {}
    for e in router:
        msgs = [Message(role=m["role"], content=m["content"]) for m in e["prompt"]]
        out = llm.generate(msgs, max_tokens=120, purpose="sft_eval").text
        parsed = parse_json_object(out) or {}
        pred = str(parsed.get("action", "")).strip().lower() or "<invalid>"
        valid += int("action" in parsed)
        correct += int(pred == e["gold_action"])
        confusion.setdefault(e["gold_action"], {}).setdefault(pred, 0)
        confusion[e["gold_action"]][pred] += 1
    n = len(router)
    return {"model": base_model, "adapter": adapter_path, "n": n,
            "json_valid_rate": valid / n if n else 0.0, "action_accuracy": correct / n if n else 0.0,
            "confusion": confusion}


def load_val_examples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
