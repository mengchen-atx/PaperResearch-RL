"""Table detection at the source: a PyMuPDF upgrade that changes find_tables() fails here,
not three layers up in the agent/RL integration tests."""

import pymupdf

from atlas.common.types import Document
from atlas.multimodal.pdf_extract import extract_tables
from tests.conftest import make_pdf


def test_extract_tables_finds_the_ruled_table(tmp_path):
    path = tmp_path / "widget.pdf"
    make_pdf(path)
    doc = Document(doc_id="d1", filename=path.name, source_path=str(path))
    with pymupdf.open(path) as pdf:
        tables = [t for page in pdf for t in extract_tables(page, doc)]
    assert len(tables) == 1, f"pymupdf {pymupdf.__version__}: expected 1 table, found {len(tables)}"
    t = tables[0]
    assert t.table_id == "Table 1" and t.page == 2
    assert t.header == ["Model", "BLEU", "Params"]
    assert ["Widget-L", "28.4", "210"] in t.rows
