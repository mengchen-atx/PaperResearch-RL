import sqlite3
import time

import pytest

from atlas.common.config import ToolsConfig
from atlas.common.errors import ToolError
from atlas.tools.base import Tool, ToolArgs, ToolExecutor, ToolResult
from atlas.tools.python_sandbox import RunPython, RunPythonArgs, validate_code
from atlas.tools.sql import LookupTable, LookupTableArgs, QuerySQL, QuerySQLArgs, validate_sql


def test_python_sandbox_runs_math():
    tool = RunPython(ToolsConfig(python_mode="sandbox"))
    res = tool.run(RunPythonArgs(code="import math\nprint(round(28.4 - 26.36, 2), math.sqrt(16))"))
    assert res.ok and res.output == "2.04 4.0" and res.evidence[0].modality == "computation"


@pytest.mark.parametrize("code", ["import os\nos.system('ls')", "open('/etc/passwd').read()",
                                  "().__class__.__bases__", "__import__('os')", "eval('1+1')",
                                  "from subprocess import run"])
def test_python_sandbox_rejects_unsafe_code(code):
    with pytest.raises(ToolError):
        validate_code(code)


def test_python_sandbox_timeout_and_disabled():
    tool = RunPython(ToolsConfig(python_mode="sandbox", python_timeout_s=1))
    with pytest.raises(ToolError, match="timed out"):
        tool.run(RunPythonArgs(code="while True:\n    pass"))
    with pytest.raises(ToolError, match="disabled"):
        RunPython(ToolsConfig(python_mode="disabled")).run(RunPythonArgs(code="print(1)"))


def test_python_runtime_error_is_reported_not_raised():
    res = RunPython(ToolsConfig()).run(RunPythonArgs(code="print(1/0)"))
    assert not res.ok and "ZeroDivisionError" in res.error


@pytest.mark.parametrize("sql", ["DROP TABLE x", "select 1; select 2", "PRAGMA table_info(x)",
                                 "ATTACH DATABASE 'x' AS y", "delete from atlas_tables"])
def test_sql_validation_rejects(sql):
    with pytest.raises(ToolError):
        validate_sql(sql)


@pytest.fixture
def table_db(tmp_path):
    db = tmp_path / "t.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE atlas_tables (table_name TEXT, table_id TEXT, doc_id TEXT, filename TEXT, "
                  "page INTEGER, caption TEXT, columns TEXT)")
        c.execute("CREATE TABLE t_1 (model, bleu)")
        c.executemany("INSERT INTO t_1 VALUES (?, ?)", [("A", 25.1), ("B", 28.4)])
        c.execute("INSERT INTO atlas_tables VALUES ('t_1','Table 2','d','p.pdf',8,'Table 2: BLEU scores','[\"model\",\"bleu\"]')")
    return db


def test_query_sql_readonly_and_results(table_db):
    res = QuerySQL(table_db).run(QuerySQLArgs(sql="SELECT model FROM t_1 ORDER BY bleu DESC LIMIT 1"))
    assert res.ok and "B" in res.output
    bad = QuerySQL(table_db).run(QuerySQLArgs(sql="SELECT * FROM missing_table"))
    assert not bad.ok and "SQL error" in bad.error


def test_lookup_table_finds_by_caption(table_db):
    res = LookupTable(table_db).run(LookupTableArgs(query="BLEU scores"))
    assert res.ok and res.evidence[0].page == 8 and "28.4" in res.evidence[0].content


class _Slow(Tool):
    name, description, Args = "slow", "sleeps", ToolArgs

    def run(self, args):
        time.sleep(2)
        return ToolResult(tool="slow", ok=True)


class _Boom(Tool):
    name, description, Args = "boom", "raises", ToolArgs

    def run(self, args):
        raise RuntimeError("kaboom")


def test_executor_handles_unknown_invalid_timeout_and_crash():
    ex = ToolExecutor([_Slow(), _Boom(), QuerySQL(__file__)], timeout_s=0.3)
    r, rec = ex.call("nope", {})
    assert not r.ok and "Unknown tool" in r.error and rec.tool == "nope"
    r, _ = ex.call("query_sql", {"sql": 1, "extra": True})
    assert not r.ok and "Invalid arguments" in r.error
    r, rec = ex.call("slow", {})
    assert not r.ok and "timed out" in r.error
    r, _ = ex.call("boom", {})
    assert not r.ok and "kaboom" in r.error
