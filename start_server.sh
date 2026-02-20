#!/usr/bin/env bash
set -euo pipefail

echo "Starting RAG Chat (v4.0.0)..."
echo ""
echo "Tip: first startup downloads the embedding model (~440MB) and caches it."
echo "     Use 'python cli.py ingest' to index your knowledge base files."
echo ""
cd "$(dirname "$0")"
.venv/bin/python cli.py dev
