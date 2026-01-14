#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Load settings from config (optional override via env vars)
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-3001}"

echo "Starting MMRAG API on http://${HOST}:${PORT}"
exec uvicorn app.api:app --host "$HOST" --port "$PORT" --reload
