"""Build + verify the Atlas papers benchmark (data/benchmarks/atlas_papers_v1.jsonl).

Every question lists gold evidence as (filename, pages, quote). This script
verifies that each quote literally occurs on every listed page of the real PDF
(whitespace-normalized) and refuses to write the benchmark otherwise, so gold
labels are checked against the source rather than trusted from memory.

Types: text | table | compute | figure | multihop
Numeric questions carry `numeric` (+ `tolerance`) and are scored numerically.

Usage: python scripts/build_benchmark.py [--raw data/raw] [--out data/benchmarks/atlas_papers_v1.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

A, B, R, S = "arxiv_1706.03762.pdf", "arxiv_1810.04805.pdf", "arxiv_2005.11401.pdf", "arxiv_2001.08361.pdf"


def q(qid, question, answer, qtype, evidence, aliases=(), numeric=None, tolerance=None, modalities=("text",)):
    item = {"id": qid, "question": question, "answer": answer, "aliases": list(aliases), "type": qtype,
            "modalities": list(modalities),
            "evidence": [{"filename": f, "pages": list(p), "quote": quote} for f, p, quote in evidence]}
    if numeric is not None:
        item["numeric"] = numeric
        item["tolerance"] = tolerance if tolerance is not None else max(abs(numeric) * 0.02, 0.01)
    return item


ITEMS = [
    # ---------------- Transformer (Attention Is All You Need) ----------------
    q("attn-01", "What is the model dimension d_model used by the Transformer?", "512", "text",
      [(A, [2], "dmodel = 512")]),
    q("attn-02", "How many parallel attention heads does the base Transformer use?", "8", "text",
      [(A, [4], "h = 8")], aliases=["eight"]),
    q("attn-03", "What is the inner-layer dimensionality of the Transformer's position-wise feed-forward networks?",
      "2048", "text", [(A, [5], "inner-layer has dimension 2048")]),
    q("attn-04", "By what factor are the dot products scaled in scaled dot-product attention?", "1/√dk", "text",
      [(A, [4], "1/√dk")], aliases=["square root of d_k", "sqrt(d_k)", "√dk", "sqrt(dk)", "square root of dk"]),
    q("attn-05", "Which functions does the Transformer use for its positional encodings?", "sine and cosine", "text",
      [(A, [5], "sine and cosine")], aliases=["sinusoid", "sin and cos", "sine and cosine functions"]),
    q("attn-06", "How long did training the big Transformer models take?", "3.5 days", "text",
      [(A, [7], "3.5 days")], aliases=["3.5"]),
    q("attn-07", "How many warmup steps were used in the Transformer's learning-rate schedule?", "4000", "text",
      [(A, [7], "warmup_steps = 4000")], aliases=["4,000"]),
    q("attn-08", "What value of beta_2 was used for the Adam optimizer when training the Transformer?", "0.98",
      "text", [(A, [7], "β2 = 0.98")]),
    q("attn-09", "What beam size was used for decoding in the Transformer machine translation experiments?", "4",
      "text", [(A, [8], "beam size of 4")], aliases=["four"]),
    q("attn-10", "What BLEU score does the big Transformer achieve on WMT 2014 English-to-German?", "28.4", "table",
      [(A, [1, 8], "28.4")], modalities=["text", "table"]),
    q("attn-11", "What BLEU score does the big Transformer achieve on WMT 2014 English-to-French?", "41.0", "table",
      [(A, [1, 8], "41.0")], aliases=["41"], modalities=["text", "table"]),
    q("attn-12", "In Table 2 of the Transformer paper, what EN-DE BLEU score does the ConvS2S Ensemble obtain?",
      "26.36", "table", [(A, [8], "26.36")], modalities=["table"]),
    q("attn-13", "What WSJ Section 23 F1 does the 4-layer Transformer reach in the semi-supervised setting?", "92.7",
      "table", [(A, [9], "92.7")], modalities=["table"]),
    q("attn-14", "How many parameters (in millions) does the big Transformer model have according to Table 3?", "213",
      "table", [(A, [9], "213")], aliases=["213M", "213 million"], modalities=["table"]),
    q("attn-15", "By how many BLEU points does Transformer (big) outperform the ConvS2S Ensemble on EN-DE?", "2.04",
      "compute", [(A, [8], "26.36"), (A, [8], "28.4")], numeric=2.04, tolerance=0.02,
      modalities=["table", "computation"]),
    q("attn-16", "How many times larger is the EN-DE training cost (FLOPs) of Transformer (big) than of the base "
      "Transformer in Table 2?", "about 7", "compute", [(A, [8], "3.3 · 1018"), (A, [8], "2.3 · 1019")],
      numeric=6.97, tolerance=0.15, modalities=["table", "computation"]),
    # ---------------- BERT ----------------
    q("bert-01", "What is the hidden size H of BERT-Base?", "768", "text", [(B, [3], "L=12, H=768, A=12")]),
    q("bert-02", "How many total parameters does BERT-Large have?", "340M", "text", [(B, [3], "340M")],
      aliases=["340 million", "340"]),
    q("bert-03", "What percentage of WordPiece tokens are masked at random in BERT's masked LM pre-training?", "15%",
      "text", [(B, [4], "15%")], aliases=["15 percent", "15"]),
    q("bert-04", "How many words does the BooksCorpus used to pre-train BERT contain?", "800M", "text",
      [(B, [5], "800M words")], aliases=["800 million", "800M words"]),
    q("bert-05", "How many words of English Wikipedia were used to pre-train BERT?", "2,500M", "text",
      [(B, [5], "2,500M words")], aliases=["2500M", "2.5B", "2.5 billion", "2,500 million"]),
    q("bert-06", "What GLUE score does BERT achieve according to the abstract?", "80.4", "text",
      [(B, [1], "80.4%")], aliases=["80.4%"]),
    q("bert-07", "What Test F1 does BERT reach on SQuAD v1.1?", "93.2", "text", [(B, [1], "93.2")]),
    q("bert-08", "What is the size of the WordPiece vocabulary used by BERT?", "30,000", "text",
      [(B, [3], "30,000")], aliases=["30000", "30k", "30K"]),
    q("bert-09", "What average GLUE score does BERT-Base obtain in Table 1?", "79.6", "table",
      [(B, [8], "79.6")], modalities=["table"]),
    q("bert-10", "How many more parameters (in millions) does BERT-Large have than BERT-Base?", "230M", "compute",
      [(B, [3], "110M"), (B, [3], "340M")], numeric=230, tolerance=1, modalities=["text", "computation"]),
    # ---------------- RAG ----------------
    q("rag-01", "Which pre-trained seq2seq model is used as the generator in RAG?", "BART-large", "text",
      [(R, [3], "BART-large")], aliases=["BART"]),
    q("rag-02", "How many parameters does RAG's BART-large generator have?", "400M", "text", [(R, [3], "400M")],
      aliases=["400 million"]),
    q("rag-03", "Which retriever does RAG use for its non-parametric memory?", "Dense Passage Retriever", "text",
      [(R, [2], "Dense Passage Retriever")], aliases=["DPR"]),
    q("rag-04", "What exact-match score does RAG-Sequence achieve on Natural Questions (NQ)?", "44.5", "table",
      [(R, [6], "44.5")], modalities=["table"]),
    q("rag-05", "How are Wikipedia articles split into documents for RAG's retrieval index?", "100-word chunks",
      "text", [(R, [4], "100-word")], aliases=["100-word", "100 word", "100 words"]),
    q("rag-06", "Which library does RAG use for Maximum Inner Product Search over the document index?", "FAISS",
      "text", [(R, [4], "FAISS")]),
    q("rag-07", "By how many exact-match points does RAG-Sequence beat DPR on NQ in Table 1?", "3.0", "compute",
      [(R, [6], "41.5"), (R, [6], "44.5")], numeric=3.0, tolerance=0.05, modalities=["table", "computation"]),
    q("rag-08", "What FEVER-3 label accuracy does RAG-Token achieve?", "72.5", "table", [(R, [6], "72.5")],
      modalities=["table"]),
    # ---------------- Scaling laws ----------------
    q("scale-01", "On which dataset are the language models in 'Scaling Laws for Neural Language Models' trained?",
      "WebText2", "text", [(S, [6, 7], "WebText2")]),
    q("scale-02", "What is the power-law exponent alpha_N for loss as a function of non-embedding parameters?",
      "0.076", "text", [(S, [4, 11, 20], "0.076")]),
    q("scale-03", "According to the scaling laws paper, when model size increases 8x, by roughly how much must the "
      "data increase to avoid a penalty?", "5x", "text", [(S, [3], "roughly 5x")], aliases=["5 times", "five"]),
    q("scale-04", "With what exponent of compute should the optimal (compute-efficient) model size N grow?", "0.73",
      "text", [(S, [5, 16, 20], "0.73")]),
    q("scale-05", "What BPE vocabulary size is used in the scaling laws experiments?", "50257", "text",
      [(S, [6], "50257")], aliases=["50,257"]),
    q("scale-06", "How many tokens does the WebText2 dataset contain?", "22B", "text",
      [(S, [7], "2.29 × 1010 tokens"), (S, [11], "22B token")], aliases=["22 billion", "2.29", "22B"]),
    # ---------------- Figure questions ----------------
    q("fig-01", "In Figure 1 of the scaling laws paper, what exponent is shown for test loss as a function of "
      "dataset size?", "0.095", "figure", [(S, [4], "0.095")], modalities=["figure"]),
    q("fig-02", "What does Figure 1 of 'Attention Is All You Need' depict?", "the Transformer model architecture",
      "figure", [(A, [3], "Figure 1: The Transformer - model architecture")],
      aliases=["model architecture", "architecture"], modalities=["figure"]),
    q("fig-03", "What does the left panel of Figure 3 in the RAG paper show?",
      "NQ performance as more documents are retrieved", "figure",
      [(R, [9], "Left: NQ performance as more documents are retrieved")],
      aliases=["more documents are retrieved", "number of retrieved documents", "documents retrieved"],
      modalities=["figure"]),
    q("fig-04", "What does Figure 2 of the BERT paper illustrate?", "BERT input representation", "figure",
      [(B, [4], "Figure 2: BERT input representation")],
      aliases=["input representation", "sum of the token embeddings"], modalities=["figure"]),
    # ---------------- Multi-hop (cross-document) ----------------
    q("multi-01", "How many more layers does BERT-Base have than the encoder stack of the base Transformer?", "6",
      "multihop", [(B, [3], "L=12"), (A, [2], "N = 6")], numeric=6, tolerance=0.01,
      modalities=["text", "computation"]),
    q("multi-02", "What is the difference between BERT-Base's hidden size and the Transformer's d_model?", "256",
      "multihop", [(B, [3], "H=768"), (A, [2], "dmodel = 512")], numeric=256, tolerance=0.5,
      modalities=["text", "computation"]),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def verify(items: list[dict], raw: Path) -> list[str]:
    pages: dict[str, list[str]] = {}
    problems = []
    for item in items:
        for ev in item["evidence"]:
            fn = ev["filename"]
            if fn not in pages:
                path = raw / fn
                if not path.exists():
                    problems.append(f"{item['id']}: missing file {fn} (run scripts/download_corpus.sh)")
                    continue
                with pymupdf.open(path) as pdf:
                    pages[fn] = [normalize(p.get_text()) for p in pdf]
            for p in ev["pages"]:
                if normalize(ev["quote"]) not in pages[fn][p - 1]:
                    problems.append(f"{item['id']}: quote {ev['quote']!r} not found on {fn} p.{p}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/benchmarks/atlas_papers_v1.jsonl")
    args = ap.parse_args()
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    problems = verify(ITEMS, Path(args.raw))
    if problems:
        print("Benchmark verification FAILED:\n  " + "\n  ".join(problems))
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    by_type: dict[str, int] = {}
    for i in ITEMS:
        by_type[i["type"]] = by_type.get(i["type"], 0) + 1
    print(f"verified {len(ITEMS)} questions against the PDFs -> {out}  {by_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
