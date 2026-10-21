import pytest

from atlas.common.config import AtlasConfig, deep_merge, env_overrides
from atlas.common.errors import UnsafePathError
from atlas.common.paths import safe_resolve
from atlas.llm.json_utils import parse_json_object


def test_env_overrides_nested_and_typed():
    env = {"ATLAS__LLM__PROVIDER": "extractive", "ATLAS__RETRIEVAL__TOP_K": "7", "OTHER": "x"}
    assert env_overrides(env) == {"llm": {"provider": "extractive"}, "retrieval": {"top_k": 7}}


def test_deep_merge_is_immutable():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def test_with_overrides_returns_new_config():
    cfg = AtlasConfig()
    new = cfg.with_overrides({"retrieval": {"top_k": 9}})
    assert new.retrieval.top_k == 9 and cfg.retrieval.top_k == 5


def test_production_forbids_python_sandbox(monkeypatch):
    monkeypatch.delenv("ATLAS_ALLOW_PYTHON_IN_PRODUCTION", raising=False)
    with pytest.raises(ValueError):
        AtlasConfig(environment="production")
    cfg = AtlasConfig().with_overrides({"environment": "production", "tools": {"python_mode": "disabled"}})
    assert cfg.environment == "production"


def test_safe_resolve_blocks_traversal(tmp_path):
    assert safe_resolve("a/b.txt", tmp_path) == (tmp_path / "a" / "b.txt").resolve()
    for bad in ("../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(UnsafePathError):
            safe_resolve(bad, tmp_path)


def test_parse_json_object_is_robust():
    assert parse_json_object('```json\n{"action": "answer",}\n```') == {"action": "answer"}
    assert parse_json_object('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None
