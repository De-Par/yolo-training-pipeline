#!/usr/bin/env sh

IS_SOURCED=0
(return 0 2>/dev/null) && IS_SOURCED=1

if [ "$IS_SOURCED" -ne 1 ]; then
    echo "[ERROR] Run this script via: source scripts/setup_env.sh"
    exit 1
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$PROJECT_ROOT" ]; then
    SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || return 1

python3 -m venv .venv || return 1
.venv/bin/python -m pip install --upgrade pip || return 1
.venv/bin/python -m pip install -r requirements.txt || return 1

echo "[INFO] Environment is ready in: $PROJECT_ROOT/.venv"
. ".venv/bin/activate" || return 1
