"""Static checks on deployment config (manifests are not applied to a cluster in CI)."""

from pathlib import Path

import yaml

from atlas.common.config import AtlasConfig, env_overrides

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "deployment" / "kubernetes"


def _docs():
    for f in sorted(K8S.glob("*.yaml")):
        for doc in yaml.safe_load_all(f.read_text()):
            if doc:
                yield f.name, doc


def test_manifests_parse_and_have_kinds():
    kinds = {d["kind"] for _, d in _docs()}
    assert {"Deployment", "Service", "StatefulSet", "HorizontalPodAutoscaler", "ConfigMap", "Ingress"} <= kinds


def test_workloads_define_resources_and_probes():
    for name, d in _docs():
        if d["kind"] not in ("Deployment", "StatefulSet"):
            continue
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert "requests" in c["resources"] and "limits" in c["resources"], f"{name}:{c['name']}"
            if c["name"] != "worker":  # workers have no HTTP endpoint
                assert "readinessProbe" in c, f"{name}:{c['name']} missing readinessProbe"


def test_production_configmap_is_a_valid_atlas_config():
    cm = next(d for _, d in _docs() if d["kind"] == "ConfigMap")
    cfg = AtlasConfig().with_overrides(env_overrides(cm["data"]))
    assert cfg.environment == "production" and cfg.tools.python_mode == "disabled"
    assert cfg.serving.queue_backend == "redis" and cfg.retrieval.vector_store == "qdrant"


def test_compose_file_parses():
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text())
    assert {"api", "worker", "redis", "qdrant", "vllm"} <= set(compose["services"])
