#!/usr/bin/env bash
# Reset the FAISS index by clearing data/mmrag/index/
# Documents are preserved; only the index is removed.

set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

INDEX_DIR="data/mmrag/index"
DOCS_DIR="data/mmrag/docs"

echo "🗑️  Resetting MMRAG index..."

# Remove index files
if [ -d "$INDEX_DIR" ]; then
    rm -rf "$INDEX_DIR"
    echo "   Removed $INDEX_DIR"
else
    echo "   $INDEX_DIR does not exist (nothing to remove)"
fi

# Recreate empty index directory
mkdir -p "$INDEX_DIR"
echo "   Created empty $INDEX_DIR"

# Mark all manifests as not indexed
if [ -d "$DOCS_DIR" ]; then
    for manifest in "$DOCS_DIR"/*/manifest.json; do
        if [ -f "$manifest" ]; then
            # Use Python to safely update JSON
            python3 -c "
import json
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    data = json.load(f)
data['indexed'] = False
data['indexed_at'] = None
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'   Updated: {path}')
" "$manifest"
        fi
    done
fi

echo "✅ Index reset complete. Run reindex_all.sh to rebuild."
