"""Persistent research history (SQLite).

Stores completed research sessions so that (a) the API can list/inspect past
sessions, and (b) a new query can surface closely related past research
(BM25 over past queries) instead of starting from zero. This is long-term
memory at the *session* level; the knowledge itself lives in the index.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from atlas.retrieval.bm25 import BM25Index

RELATED_MIN_SCORE = 2.0


class ResearchHistory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, created REAL, "
                      "query TEXT, answer TEXT, citations TEXT, num_tool_calls INTEGER, status TEXT)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record(self, session_id: str, query: str, answer: str, citations: list[dict], num_tool_calls: int,
               status: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?)",
                      (session_id, time.time(), query, answer, json.dumps(citations), num_tool_calls, status))

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT session_id, created, query, answer, status FROM sessions "
                             "ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"session_id": r[0], "created": r[1], "query": r[2], "answer": r[3], "status": r[4]} for r in rows]

    def related(self, query: str, limit: int = 3) -> list[dict]:
        rows = self.recent(limit=500)
        if not rows:
            return []
        scores = BM25Index().build([r["query"] for r in rows]).scores(query)
        order = scores.argsort()[::-1][:limit]
        return [{**rows[i], "score": float(scores[i])} for i in order if scores[i] >= RELATED_MIN_SCORE]
