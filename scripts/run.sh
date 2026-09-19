#!/usr/bin/env bash
# Start the API (and serve the UI) on http://127.0.0.1:8077
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run uvicorn library_agent.api.app:app --host 127.0.0.1 --port "${PORT:-8077}" --reload
