#!/usr/bin/env bash
# Download the public demo corpus (arXiv v1 PDFs) used by the benchmark.
# The benchmark's gold page numbers refer to these exact versions (v1).
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "$DEST"
for id in 1706.03762 1810.04805 2005.11401 2001.08361; do
  out="$DEST/arxiv_${id}.pdf"
  if [ ! -s "$out" ]; then
    echo "downloading arXiv:${id}v1"
    curl -fsSL -o "$out" "https://arxiv.org/pdf/${id}v1"
  fi
done
echo "corpus ready in $DEST:"; ls -la "$DEST"/*.pdf
