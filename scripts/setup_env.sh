#!/usr/bin/env sh

IS_SOURCED=0
(return 0 2>/dev/null) && IS_SOURCED=1

if [ "$IS_SOURCED" -ne 1 ]; then
    echo "[ERROR] Run this script via: source scripts/setup_env.sh [base|cpu|cuda]"
    exit 1
fi

if [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_SOURCE="${(%):-%N}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
else
    SCRIPT_SOURCE="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    echo "[ERROR] Could not resolve project root from: ${SCRIPT_SOURCE}" >&2
    echo "[ERROR] Resolved PROJECT_ROOT='${PROJECT_ROOT}', but pyproject.toml was not found there." >&2
    return 1
fi

cd "${PROJECT_ROOT}" || return 1

PROFILE="base"

for arg in "$@"; do
    case "$arg" in
        base)
            PROFILE="base"
            ;;
        cpu)
            PROFILE="cpu"
            ;;
        cuda)
            PROFILE="cuda"
            ;;
        *)
            echo "[ERROR] Unknown option: $arg" >&2
            echo "[INFO] Usage: source scripts/setup_env.sh [base|cpu|cuda]" >&2
            return 1
            ;;
    esac
done

python3 -m venv .venv || return 1
.venv/bin/python -m pip install --upgrade pip setuptools wheel || return 1

EXTRAS=""
if [ "$PROFILE" = "cpu" ]; then
    EXTRAS="cpu"
elif [ "$PROFILE" = "cuda" ]; then
    EXTRAS="cuda"
fi

if [ -n "$EXTRAS" ]; then
    INSTALL_TARGET=".[${EXTRAS}]"
else
    INSTALL_TARGET="."
fi

echo "[INFO] PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[INFO] PROFILE: ${PROFILE}"
echo "[INFO] Installing: ${INSTALL_TARGET}"

.venv/bin/python -m pip install -e "${INSTALL_TARGET}" --no-build-isolation || return 1

echo "[INFO] Environment is ready in: ${PROJECT_ROOT}/.venv"
. "${PROJECT_ROOT}/.venv/bin/activate" || return 1
