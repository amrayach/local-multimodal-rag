#!/usr/bin/env bash
# Rebuild the FAISS index from all existing documents.
# This walks data/mmrag/docs/ and re-embeds all pages.

set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

echo "🔄 Reindexing all documents..."

python3 -c "
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

from app.rag.pipeline import MMRagPipeline

# Initialize pipeline
pipe = MMRagPipeline()

# Clear existing index first
pipe.clear_index()

# Reindex all documents
result = pipe.reindex_all()

print()
print(f'✅ Reindex complete:')
print(f'   Documents: {result[\"docs\"]}')
print(f'   Pages: {result[\"pages\"]}')
print(f'   Elapsed: {result[\"elapsed_ms\"]}ms')
"
