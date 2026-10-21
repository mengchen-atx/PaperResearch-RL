"""agent -> tools -> evidence -> critic -> writer, through the real LangGraph graph."""

from atlas.agents.graph import ResearchAgent
from atlas.agents.nodes import AgentContext
from atlas.agents.policies import LLMPolicy
from atlas.trajectory.store import validate_trajectory
from tests.conftest import FakeVLM
from tests.unit.test_agent_units import FakeLLM


def test_agent_runs_tools_and_answers_with_citations(built_runtime):
    res = built_runtime.agent().run("What BLEU score does Widget-L get in Table 1?")
    tools = [r.tool for r in res.tool_history]
    assert tools[0] == "search_documents" and "lookup_table" in tools
    assert res.status == "answered" and "28.4" in res.answer and res.citations
    assert any(e.modality == "table" for e in res.evidence)
    traj = res.to_trajectory()
    assert validate_trajectory(traj) == [] and traj.steps[-1].action == "answer"
    assert res.trace_id and any(c["node"] == "writer" for c in res.llm_calls)


def test_agent_uses_vlm_for_figure_questions(runtime_factory):
    rt = runtime_factory()
    fake = FakeVLM()
    rt.__dict__["vlm"] = fake
    rt.cfg = rt.cfg.with_overrides({"vlm": {"provider": "hf_local"}})  # enable the visual tool (fake model)
    rt.__dict__.pop("executor", None)
    res = rt.agent().run("What does the bar chart in Figure 1 show about widget accuracy?")
    assert "analyze_image" in [r.tool for r in res.tool_history] and fake.calls == 1
    assert any(e.modality == "figure" and "VLM reading" in e.content for e in res.evidence)


def test_budget_limits_are_enforced(runtime_factory):
    rt = runtime_factory(agent={"max_iterations": 1, "max_tool_calls": 1})
    res = rt.agent().run("What BLEU score does Widget-L get in Table 1?")
    assert len(res.tool_history) <= 1 and res.status in ("budget_exhausted", "answered") and res.answer


def test_llm_policy_with_invalid_output_falls_back(built_runtime):
    ctx = AgentContext(cfg=built_runtime.cfg.agent, llm=built_runtime.llm, executor=built_runtime.executor,
                       policy=LLMPolicy(FakeLLM("not json at all")), caps=built_runtime.capabilities())
    res = ResearchAgent(ctx).run("What is the capital of Zedland?")
    assert res.status == "answered" and all(d["fallback"] for d in res.decisions if d["policy"] == "llm")
    assert "Quorra" in res.answer


def test_failing_tool_is_recorded_not_fatal(runtime_factory):
    rt = runtime_factory()

    class Broken:
        name = "lookup_table"

        def run(self, args):
            raise RuntimeError("database offline")

    rt.executor.tools["lookup_table"].run = Broken().run
    res = rt.agent().run("What BLEU score does Widget-L get in Table 1?")
    failed = [r for r in res.tool_history if not r.ok]
    assert failed and "database offline" in failed[0].error and res.answer


def test_evaluated_trajectories_keep_writer_evidence_ids(built_runtime, tmp_path):
    """Regression: the eval runner used to replace trajectory metadata, dropping the ids
    needed to rebuild writer SFT examples."""
    from atlas.evaluation.benchmark import BenchmarkItem
    from atlas.evaluation.runner import evaluate_agent
    from atlas.training.sft_data import writer_example
    from atlas.trajectory.store import TrajectoryStore

    store = TrajectoryStore(tmp_path / "t.jsonl")
    item = BenchmarkItem(id="z", question="What is the capital of Zedland?", answer="Quorra")
    evaluate_agent(built_runtime.agent(), [item], store)
    traj = store.load()[0]
    assert traj.metadata["benchmark_id"] == "z" and traj.metadata["writer_evidence_ids"]
    assert writer_example(traj) is not None
