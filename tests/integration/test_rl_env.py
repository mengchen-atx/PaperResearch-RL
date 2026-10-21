import numpy as np

from atlas.agents.actions import ACTIONS, Action
from atlas.evaluation.benchmark import BenchmarkItem, GoldEvidence
from atlas.llm.providers import ExtractiveLLM
from atlas.rl.env import ResearchEnv
from atlas.rl.policy import PolicyNetwork
from atlas.rl.train import RLTrainConfig, evaluate_policy, heuristic_chooser, train

ITEM = BenchmarkItem(id="z", question="What is the capital of Zedland?", answer="Quorra",
                     evidence=[GoldEvidence(filename="facts.txt", pages=[])])
TABLE_ITEM = BenchmarkItem(id="t", question="What BLEU score does Widget-L get in Table 1?", answer="28.4",
                           evidence=[GoldEvidence(filename="widget_paper.pdf", pages=[2])])


def make_env(rt):
    return ResearchEnv(rt.executor, rt.capabilities(), rt.cfg.agent, ExtractiveLLM())


def test_env_step_reset_and_termination(built_runtime):
    env = make_env(built_runtime)
    obs, mask = env.reset(TABLE_ITEM)
    assert obs.shape[0] == PolicyNetwork().obs_dim and mask[ACTIONS.index(Action.ANSWER)]
    out = env.step(ACTIONS.index(Action.SEARCH_DOCUMENTS))
    assert not out.terminated and out.info["reward_components"]["evidence_shaping"] > 0
    out = env.step(ACTIONS.index(Action.ANSWER))
    assert out.terminated and out.info["reward_components"]["task_success"] == 1.0


def test_env_truncates_at_budget(built_runtime):
    env = make_env(built_runtime)
    env.reset(ITEM)
    done, steps = False, 0
    while not done:
        out = env.step(ACTIONS.index(Action.SEARCH_DOCUMENTS))
        done, steps = out.terminated or out.truncated, steps + 1
    assert out.truncated and steps <= built_runtime.cfg.agent.max_iterations


def test_policy_masking_never_samples_invalid_actions():
    net = PolicyNetwork()
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[ACTIONS.index(Action.ANSWER)] = True
    assert {net.act(np.zeros(net.obs_dim, dtype=np.float32), mask)[0] for _ in range(20)} == {
        ACTIONS.index(Action.ANSWER)}


def test_training_loop_runs_and_saves(built_runtime, tmp_path):
    env = make_env(built_runtime)
    res = train(env, [ITEM, TABLE_ITEM], [TABLE_ITEM], RLTrainConfig(iterations=2, episodes_per_update=2), tmp_path)
    assert (tmp_path / "policy.pt").exists() and len(res["history"]) == 2
    assert set(res["test"]) == {"learned_greedy", "heuristic", "random"}
    loaded = PolicyNetwork.load(tmp_path / "policy.pt")
    assert loaded.n_actions == len(ACTIONS)
    assert evaluate_policy(env, [ITEM], heuristic_chooser(env))["success_rate"] == 1.0
