#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Optional env var overrides (defaults match config.py)
export UI_HOST="${UI_HOST:-0.0.0.0}"
export UI_PORT="${UI_PORT:-8081}"
export API_BASE="${API_BASE:-http://127.0.0.1:3001}"

echo "Starting MMRAG UI on http://${UI_HOST}:${UI_PORT}"
echo "Connecting to API at ${API_BASE}"
exec python ui/gradio_app.py
