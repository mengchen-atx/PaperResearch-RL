"""API request -> agent, using FastAPI's TestClient against the offline runtime."""

import time

import fakeredis
import pytest
from fastapi.testclient import TestClient

from atlas.serving.api import create_app
from atlas.serving.jobs import RedisJobStore


@pytest.fixture(scope="module")
def client(built_runtime):
    app = create_app(built_runtime.cfg, runtime=built_runtime)
    with TestClient(app) as c:
        yield c


def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    r = client.get("/metrics")
    assert r.status_code == 200 and "atlas_requests_total" in r.text


def test_retrieve_and_validation(client):
    r = client.post("/retrieve", json={"query": "capital of Zedland", "top_k": 3})
    assert r.status_code == 200 and r.json()["results"][0]["filename"] == "facts.txt"
    assert r.headers["X-Request-ID"]
    assert client.post("/retrieve", json={"query": "", "top_k": 3}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 999}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "unknown": 1}).status_code == 422


def test_ask_returns_cited_answer(client):
    body = client.post("/ask", json={"question": "What is the capital of Zedland?"}).json()
    assert "Quorra" in body["answer"] and body["citations"] and body["invalid_citations"] == []


def test_research_sync(client):
    r = client.post("/research", json={"query": "What is the capital of Zedland?", "mode": "sync"})
    assert r.status_code == 200 and "Quorra" in r.json()["answer"]
    spans = client.get(f"/traces/{r.json()['trace_id']}").json()
    assert {s["kind"] for s in spans} >= {"request", "agent_node", "tool"}


def test_research_async_poll_events_and_idempotency(client):
    headers = {"Idempotency-Key": "abc-123"}
    r = client.post("/research", json={"query": "What BLEU score does Widget-L get in Table 1?"}, headers=headers)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert client.post("/research", json={"query": "What BLEU score does Widget-L get in Table 1?"},
                       headers=headers).json()["job_id"] == job_id
    for _ in range(100):
        job = client.get(f"/research/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "succeeded" and "28.4" in job["result"]["answer"]
    stream = client.get(f"/research/{job_id}/events").text
    assert "event: planner" in stream and "event: act" in stream and "event: end" in stream
    assert client.get("/research/job_missing").status_code == 404
    assert any(s["session_id"] == job_id for s in client.get("/sessions").json())


def test_upload_validation_and_reindex(runtime_factory):
    # Own runtime: an upload triggers a re-index, which must not disturb the shared fixture.
    rt = runtime_factory()
    with TestClient(create_app(rt.cfg, runtime=rt)) as c:
        assert c.post("/documents", files={"file": ("x.exe", b"MZ")}).status_code == 415
        r = c.post("/documents", files={"file": ("../../evil.txt", b"Uploaded note: the Orion widget is blue.")})
        assert r.status_code == 202 and r.json()["saved"] == "evil.txt"  # path components stripped
        upload_dir = rt.cfg.resolve_path(rt.cfg.paths.upload_dir)
        assert (upload_dir / "evil.txt").exists() and not (upload_dir.parent.parent / "evil.txt").exists()
        for _ in range(100):
            if c.get("/documents/status").json()["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert c.get("/documents/status").json()["state"] == "done"
        assert c.post("/retrieve", json={"query": "Orion widget color"}).json()["results"][0]["filename"] == "evil.txt"


def test_rate_limit(built_runtime):
    cfg = built_runtime.cfg.with_overrides({"serving": {"rate_limit_per_minute": 2}})
    with TestClient(create_app(cfg, runtime=built_runtime)) as c:
        codes = [c.post("/retrieve", json={"query": "bm25"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_redis_job_backend_with_worker_semantics(built_runtime):
    from atlas.serving.jobs import run_research_job

    store = RedisJobStore(fakeredis.FakeRedis())
    job, _ = store.create("What is the capital of Zedland?", None)
    store.enqueue(job.job_id)
    job_id = store.dequeue(1)
    done = run_research_job(built_runtime, store, job_id)
    assert done.status == "succeeded" and "Quorra" in done.result["answer"]
    assert run_research_job(built_runtime, store, job_id).status == "succeeded"  # redelivery is idempotent
