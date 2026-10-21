import fakeredis
import pytest

from atlas.agents.actions import Action
from atlas.common.types import Citation, Evidence, Modality
from atlas.evaluation.benchmark import BenchmarkItem, GoldEvidence
from atlas.rl.reward import RewardConfig, evidence_potential, step_reward, terminal_reward
from atlas.rl.train import reward_to_go
from atlas.serving.jobs import MemoryJobStore, RedisJobStore
from atlas.training.sft_data import router_examples, split_by_trajectory, writer_example
from atlas.trajectory.schema import StepObservation, Trajectory, TrajectoryStep
from atlas.trajectory.store import TrajectoryStore, quality_filter, validate_trajectory


def make_traj(tid="t1", success=True):
    evs = [Evidence(evidence_id=f"e{i}", modality=Modality.TEXT, content=f"fact {i}", source_tool="s",
                    filename="a.pdf", page=i) for i in range(3)]
    steps = [
        TrajectoryStep(index=0, action="search_documents", tool="search_documents", arguments={"query": "q"},
                       observation=StepObservation(ok=True, new_evidence=2)),
        TrajectoryStep(index=1, action="lookup_table", tool="lookup_table", arguments={"query": "q"},
                       observation=StepObservation(ok=True, new_evidence=1)),
        TrajectoryStep(index=2, action="answer"),
    ]
    return Trajectory(trajectory_id=tid, query="What is q?", steps=steps, evidence=evs, final_answer="q is 1 [1].",
                      citations=[Citation(marker=1, evidence_id="e0", label="a.pdf, p.0")], success=success,
                      evaluation={"answer_score": 1.0}, metadata={"writer_evidence_ids": ["e0", "e2"]})


def test_trajectory_roundtrip_and_validation(tmp_path):
    store = TrajectoryStore(tmp_path / "t.jsonl")
    original = make_traj()
    store.append(original)
    loaded = store.load()
    assert loaded[0] == original and validate_trajectory(loaded[0]) == []
    bad = make_traj().model_copy(update={"final_answer": "", "citations": [Citation(marker=1, evidence_id="zz", label="")]})
    problems = validate_trajectory(bad)
    assert any("empty final answer" in p for p in problems) and any("unknown evidence" in p for p in problems)


def test_quality_filter():
    keep = quality_filter()
    assert keep(make_traj()) and not keep(make_traj(success=False))


def test_router_examples_rebuild_evidence_prefix():
    exs = router_examples(make_traj())
    assert [e["gold_action"] for e in exs] == ["search_documents", "lookup_table", "answer"]
    assert "(none yet)" in exs[0]["prompt"][1]["content"]
    assert "fact 1" in exs[1]["prompt"][1]["content"] and "fact 2" not in exs[1]["prompt"][1]["content"]
    assert "fact 2" in exs[2]["prompt"][1]["content"]
    w = writer_example(make_traj())
    assert w["completion"][0]["content"] == "q is 1 [1]." and "fact 2" in w["prompt"][1]["content"]


def test_split_by_trajectory_has_no_leakage():
    exs = router_examples(make_traj("a")) + router_examples(make_traj("b")) + router_examples(make_traj("c"))
    train, val = split_by_trajectory(exs, 0.34, 0)
    assert {e["trajectory_id"] for e in train}.isdisjoint({e["trajectory_id"] for e in val}) and val


def test_reward_components_and_shaping_telescopes():
    item = BenchmarkItem(id="x", question="q", answer="28.4", evidence=[GoldEvidence(filename="a.pdf", pages=[8])])
    cfg = RewardConfig()
    evs = [Evidence(evidence_id="e", modality=Modality.TEXT, content="c", source_tool="s", filename="a.pdf", page=8)]
    assert evidence_potential([], item) == 0.0 and evidence_potential(evs, item) == 1.0
    r = step_reward(cfg, Action.ANALYZE_VISUAL, 0.0, 1.0, invalid=False)
    assert r["evidence_shaping"] == pytest.approx(cfg.gamma_evidence_quality)
    assert r["tool_cost"] == pytest.approx(-cfg.lambda_tool_cost * 3.0)
    t = terminal_reward(cfg, "It is 28.4 [1].", item, [("a.pdf", 8)])
    assert t["task_success"] == cfg.alpha_task_success and t["citation_quality"] == cfg.beta_citation_quality
    assert reward_to_go([1.0, 0.0, 2.0], 1.0) == [3.0, 2.0, 2.0]


@pytest.mark.parametrize("store", [MemoryJobStore(), RedisJobStore(fakeredis.FakeRedis())])
def test_job_store_lifecycle_and_idempotency(store):
    job, created = store.create("question?", "heuristic", idempotency_key="k1")
    again, created2 = store.create("question?", "heuristic", idempotency_key="k1")
    assert created and not created2 and again.job_id == job.job_id
    store.enqueue(job.job_id)
    assert store.dequeue(1.0) == job.job_id
    store.append_event(job.job_id, {"node": "planner"})
    store.append_event(job.job_id, {"node": "router"})
    assert [e["node"] for e in store.events(job.job_id, 1)] == ["router"]
    assert store.update(job.job_id, status="succeeded").status == "succeeded"
    assert store.get("missing") is None
