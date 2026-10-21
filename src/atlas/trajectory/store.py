"""Trajectory persistence (JSONL), validation, and quality filtering."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from atlas.trajectory.schema import Trajectory

_WRITE_LOCK = threading.Lock()


class TrajectoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, traj: Trajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = traj.model_dump_json() + "\n"
        with _WRITE_LOCK, self.path.open("a") as f:
            f.write(line)

    def write_all(self, trajs: Iterable[Trajectory]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with _WRITE_LOCK, self.path.open("w") as f:
            for t in trajs:
                f.write(t.model_dump_json() + "\n")
                n += 1
        return n

    def iter(self) -> Iterator[Trajectory]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield Trajectory.model_validate_json(line)

    def load(self) -> list[Trajectory]:
        return list(self.iter())


def validate_trajectory(t: Trajectory) -> list[str]:
    """Structural + consistency checks. Returns a list of problems (empty == valid)."""
    problems: list[str] = []
    if not t.query.strip():
        problems.append("empty query")
    if [s.index for s in t.steps] != list(range(len(t.steps))):
        problems.append("step indices are not 0..n-1")
    if not t.steps or t.steps[-1].action != "answer":
        problems.append("trajectory does not terminate with an 'answer' action")
    if not t.final_answer.strip():
        problems.append("empty final answer")
    evidence_ids = {e.evidence_id for e in t.evidence}
    dangling = [c.evidence_id for c in t.citations if c.evidence_id not in evidence_ids]
    if dangling:
        problems.append(f"citations reference unknown evidence: {dangling[:3]}")
    for s in t.steps:
        if s.tool and s.observation is None:
            problems.append(f"step {s.index} called {s.tool} without an observation")
    return problems


def validate_file(path: Path) -> dict:
    total, valid, errors = 0, 0, {}
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                t = Trajectory.model_validate_json(line)
            except ValidationError as exc:
                errors[f"line {lineno}"] = [f"schema: {exc.error_count()} errors"]
                continue
            problems = validate_trajectory(t)
            if problems:
                errors[t.trajectory_id] = problems
            else:
                valid += 1
    return {"total": total, "valid": valid, "invalid": total - valid, "errors": errors}


def quality_filter(
    min_score: float = 0.5, require_success: bool = True, require_citations: bool = True, max_steps: int = 12
) -> Callable[[Trajectory], bool]:
    """Predicate selecting trajectories good enough to imitate (SFT data)."""

    def keep(t: Trajectory) -> bool:
        if validate_trajectory(t):
            return False
        if require_success and not t.success:
            return False
        if require_citations and not t.citations:
            return False
        if len(t.steps) > max_steps:
            return False
        score = (t.evaluation or {}).get("answer_score")
        return score is None or score >= min_score

    return keep
