"""CLI end-to-end in a subprocess: ingest -> ask -> research -> retrieve (offline providers)."""

import json
import os
import subprocess
import sys

import pytest


def run(args, env):
    return subprocess.run([sys.executable, "-m", "atlas", *args], capture_output=True, text=True, env=env, timeout=300)


@pytest.fixture
def env(tmp_path):
    e = dict(os.environ)
    e.update({"ATLAS__EMBEDDING__PROVIDER": "hashing", "ATLAS__EMBEDDING__CACHE": "false",
              "ATLAS__LLM__PROVIDER": "extractive", "ATLAS__RERANKER__ENABLED": "false",
              "ATLAS__VLM__PROVIDER": "none", "ATLAS__AGENT__PLANNER": "heuristic",
              "ATLAS__RETRIEVAL__VECTOR_STORE": "numpy",
              "ATLAS__PATHS__INDEX_DIR": str(tmp_path / "index"),
              "ATLAS__TOOLS__SQL_DB_PATH": str(tmp_path / "index" / "tables.sqlite"),
              "ATLAS__PATHS__TRAJECTORIES_DIR": str(tmp_path / "traj"),
              "ATLAS__PATHS__DATA_DIR": str(tmp_path / "data")})
    return e


def test_cli_ingest_ask_research(env, corpus_dir, tmp_path):
    r = run(["ingest", str(corpus_dir)], env)
    assert r.returncode == 0, r.stderr
    assert "3 documents" in r.stdout

    r = run(["ask", "What is the capital of Zedland?", "--json"], env)
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert "Quorra" in body["answer"] and body["retrieved"][0]["chunk"]["filename"] == "facts.txt"

    r = run(["research", "What BLEU score does Widget-L get in Table 1?"], env)
    assert r.returncode == 0, r.stderr
    assert "[tool]    lookup_table" in r.stdout and "28.4" in r.stdout
    traj = tmp_path / "traj" / "trajectories.jsonl"
    assert traj.exists()
    r = run(["trajectories", "validate", str(traj)], env)
    assert json.loads(r.stdout)["valid"] == 1


def test_cli_reports_missing_index(env):
    r = run(["ask", "anything"], env)
    assert r.returncode == 2 and "atlas ingest" in r.stderr


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("ATLAS_RUN_SLOW") != "1", reason="set ATLAS_RUN_SLOW=1 to run real-model e2e")
def test_real_models_ask():
    r = subprocess.run([sys.executable, "-m", "atlas", "ask", "What is self-attention?", "--json"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["citations"]
