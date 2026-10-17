#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi

cleanup() {
  kill "${BACK_PID:-}" "${FRONT_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python -m uvicorn autocrown.web:app --host 127.0.0.1 --port 8000 --reload &
BACK_PID=$!
(cd frontend && npm run dev) &
FRONT_PID=$!

echo "AUTO_YangGuan frontend: http://127.0.0.1:5173"
echo "AUTO_YangGuan backend : http://127.0.0.1:8000"
wait
