#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt

echo "[INFO] Environment is ready."
echo "[INFO] To activate later: source .venv/bin/activate"
