#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$HOME/prediqt/activate"
fi

export ZOLTAN="${ZOLTAN:-$HOME/prediqt/zoltan}"
export LD_LIBRARY_PATH="$ZOLTAN/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

python -m pytest -q pysph/base/tests/test_warp_nnps.py
