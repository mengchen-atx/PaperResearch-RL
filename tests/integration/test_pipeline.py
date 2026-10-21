"""ingestion -> index -> retrieval -> RAG, on the synthetic corpus (offline components)."""

from pathlib import Path

from atlas.common.types import Modality
from atlas.retrieval.filters import MetadataFilter
from atlas.retrieval.index import load_index
from atlas.tools.sql import table_catalog


def test_ingestion_extracts_text_tables_and_figures(built_runtime):
    idx = built_runtime.index
    mods = {c.modality for c in idx.chunks}
    assert mods == {Modality.TEXT, Modality.TABLE, Modality.FIGURE}
    table = next(c for c in idx.chunks if c.modality == Modality.TABLE)
    assert table.page == 2 and table.metadata["table_id"] == "Table 1" and "28.4" in table.text
    fig = next(c for c in idx.chunks if c.modality == Modality.FIGURE)
    assert fig.page == 3 and Path(fig.metadata["image_path"]).exists()
    md = [c for c in idx.chunks if c.filename == "notes.md"]
    assert {c.section for c in md} == {"Background", "Dense retrieval"}


def test_tables_are_queryable_in_sqlite(built_runtime):
    catalog = table_catalog(built_runtime.table_db)
    assert catalog and catalog[0]["columns"] == ["model", "bleu", "params"]


def test_index_roundtrip(built_runtime):
    again = load_index(built_runtime.cfg, built_runtime.index_dir)
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in built_runtime.index.chunks]


def test_hybrid_retrieval_and_metadata_filter(built_runtime):
    r = built_runtime.retriever
    hits = r.retrieve("capital of Zedland")
    assert hits[0].chunk.filename == "facts.txt" and "fused" in hits[0].component_scores
    only_tables = r.retrieve("widget results", flt=MetadataFilter(modalities=[Modality.TABLE]))
    assert only_tables and all(h.chunk.modality == Modality.TABLE for h in only_tables)


def test_rag_answer_has_valid_citations(built_runtime):
    res = built_runtime.rag().answer("What is the capital of Zedland?")
    assert "Quorra" in res.answer and res.citations and not res.invalid_citations
    assert res.citations[0].label.startswith("facts.txt")


def test_malformed_file_is_skipped_not_fatal(tmp_path, corpus_dir):
    from atlas.runtime import AtlasRuntime
    from tests.conftest import offline_config

    bad_dir = tmp_path / "docs"
    bad_dir.mkdir()
    (bad_dir / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    (bad_dir / "ok.txt").write_text("Valid document about retrieval augmented generation systems.")
    rt = AtlasRuntime(offline_config(tmp_path))
    report = rt.ingest([bad_dir])
    assert report.documents == 1 and "broken.pdf" in report.failed
