#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

STREAMLIT_HOST="0.0.0.0"
STREAMLIT_PORT="${PORT:-8501}"

export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ADDRESS="$STREAMLIT_HOST"
export STREAMLIT_SERVER_PORT="$STREAMLIT_PORT"
export STREAMLIT_BROWSER_GATHERUSAGESTATS="false"

API_PORT="8000"

uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" &
UVICORN_PID=$!

cleanup() {
  echo "Shutting down services..."
  kill "$UVICORN_PID" 2>/dev/null || true
}

trap cleanup TERM INT EXIT

echo "Starting Streamlit UI on port $STREAMLIT_PORT (Render will expose this port)."
streamlit run ui/app.py
