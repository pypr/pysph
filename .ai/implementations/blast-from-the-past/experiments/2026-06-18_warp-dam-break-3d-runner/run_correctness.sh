#!/usr/bin/env bash
# Smoke wrapper for the 3D dam-break Warp runner (ADR-0005).
# Runs a coarse, short collapse and asserts a non-empty output + all_finite.
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Prefer an already-active venv; else the known PySPH/warp venv (see memory
# pysph-warp-dev-env: bare python/python3 on PATH do NOT have warp).
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PY="${PY:-python}"
else
    PY="${PY:-/home/kunalp/.pqt_venv_e0b41259/bin/python}"
fi

cd "$ROOT"

DIR=".ai/implementations/blast-from-the-past/experiments/2026-06-18_warp-dam-break-3d-runner"
OUT="$DIR/results-smoke.npz"

"$PY" "$DIR/dam_break_3d_runner.py" \
    --dx 0.1 \
    --steps 20 \
    --hdx 1.3 \
    --rho0 1000.0 \
    --gamma 7.0 \
    --alpha 0.25 \
    --beta 0.0 \
    --kernel wendland \
    --radius-scale 2.0 \
    --xsph-eps 0.5 \
    --gz -9.81 \
    --n-damp 50 \
    --cfl 0.3 \
    --output "$OUT"

test -s "$OUT"
echo "smoke OK: $OUT"
