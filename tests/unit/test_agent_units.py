import time

from atlas.agents.actions import ACTIONS, Action, query_cues, term_coverage
from atlas.agents.features import FEATURE_DIM, Capabilities, action_mask, featurize
from atlas.agents.nodes import heuristic_plan, select_evidence
from atlas.agents.policies import HeuristicPolicy, LLMPolicy
from atlas.agents.state import initial_state
from atlas.common.types import Evidence, Modality
from atlas.llm.base import LLMClient, LLMResponse
from atlas.memory.session import call_signature, merge_evidence

CAPS = Capabilities(python=True, visual=True, web=False, tables=True, llm_can_reason=True)


def ev(i, mod=Modality.TEXT, content=None, sq=0):
    return Evidence(evidence_id=f"e{i}", modality=mod, content=content or f"content {i} with 12 and 34",
                    source_tool="t", filename="f.pdf", page=i, metadata={"subquestion": sq})


def state(query="What BLEU score does the model get in Table 2?", **kw):
    s = initial_state(query, "s", time.time(), 60)
    s.update(subquestions=[query], **kw)
    return s


def test_query_cues_and_coverage():
    assert query_cues("What does Figure 1 show?")["figure"]
    assert query_cues("By how much higher is the BLEU score?")["compute"]
    assert term_coverage("transformer bleu", ["the Transformer has BLEU 28"]) == 1.0


def test_features_fixed_dim_and_mask():
    s = state(evidence=[ev(1)])
    assert featurize(s, 6, 8).shape == (FEATURE_DIM,)
    mask = dict(zip(ACTIONS, action_mask(s, CAPS), strict=True))
    assert mask[Action.ANSWER] and mask[Action.RUN_PYTHON] and not mask[Action.SEARCH_WEB]
    no_numbers = state(evidence=[ev(1, content="no digits here")])
    assert not dict(zip(ACTIONS, action_mask(no_numbers, CAPS), strict=True))[Action.RUN_PYTHON]


def test_heuristic_policy_sequence():
    p = HeuristicPolicy()
    s = state()
    assert p.decide(s, CAPS).action == Action.SEARCH_DOCUMENTS
    s = state(searched=[call_signature("search_documents", {"query": s["query"]})], evidence=[ev(1)],
              decisions=[{"action": "search_documents"}])
    assert p.decide(s, CAPS).action == Action.LOOKUP_TABLE
    s["decisions"] = [{"action": "search_documents"}, {"action": "lookup_table"}]
    s["sufficient"] = True
    assert p.decide(s, CAPS).action == Action.ANSWER


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, text):
        super().__init__("fake")
        self.text = text

    def _generate(self, messages, max_tokens, temperature):
        return LLMResponse(text=self.text, model="fake")


def test_llm_policy_valid_and_fallback():
    s = state()
    d = LLMPolicy(FakeLLM('{"action": "lookup_table", "args": {"query": "BLEU"}, "reason": "table"}')).decide(s, CAPS)
    assert d.action == Action.LOOKUP_TABLE and d.args == {"query": "BLEU"} and not d.fallback
    d = LLMPolicy(FakeLLM("I think we should search")).decide(s, CAPS)
    assert d.fallback and d.action == Action.SEARCH_DOCUMENTS
    d = LLMPolicy(FakeLLM('{"action": "search_web"}')).decide(s, CAPS)  # web unavailable -> fallback
    assert d.fallback


def test_merge_evidence_dedupes_by_id_and_content():
    a = [ev(1)]
    merged = merge_evidence(a, [ev(1), ev(2), ev(3, content="content 1 with 12 and 34")])
    assert [e.evidence_id for e in merged] == ["e1", "e2"] and len(a) == 1


def test_heuristic_plan_splits_multi_part_questions():
    assert len(heuristic_plan("What is BM25? How does dense retrieval work?", 3)) == 2
    assert heuristic_plan("What is BM25?", 3) == ["What is BM25?"]


def test_select_evidence_prioritizes_special_modalities():
    pool = [ev(i, sq=i % 2) for i in range(10)] + [ev(99, Modality.TABLE), ev(98, Modality.COMPUTATION)]
    chosen = select_evidence(pool, 6)
    assert chosen[0].modality == Modality.COMPUTATION and len(chosen) == 6
    assert {e.metadata["subquestion"] for e in chosen if e.modality == Modality.TEXT} == {0, 1}
