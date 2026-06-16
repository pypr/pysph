#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$HOME/prediqt/activate"
fi

export ZOLTAN="${ZOLTAN:-$HOME/prediqt/zoltan}"
export LD_LIBRARY_PATH="$ZOLTAN/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

OUT=".ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-smoke.npz"

python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py \
    --nx 8 \
    --steps 2 \
    --dt 1.0e-5 \
    --rho0 1.0 \
    --c0 20.0 \
    --p0 0.0 \
    --alpha 0.1 \
    --beta 0.0 \
    --output "$OUT"

test -s "$OUT"
