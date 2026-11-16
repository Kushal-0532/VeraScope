#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/.venvs/ai/bin/python}"
exec "$PY" -m pytest -q "$@"
