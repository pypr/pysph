#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$HOME/prediqt/activate"
fi

export ZOLTAN="${ZOLTAN:-$HOME/prediqt/zoltan}"
export LD_LIBRARY_PATH="$ZOLTAN/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

python -m pytest -q pysph/base/tests/test_warp_device_helper.py

python - <<'PY'
import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
import pytest

raise SystemExit(pytest.main([
    "-q",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_constructor",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_align_particles",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_add_property",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_that_constants_can_be_added",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_remove_particles",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_add_particles",
    "pysph/base/tests/test_particle_array.py::ParticleArrayTestCPU::test_extract_particles_works_without_specific_props_without_dest",
]))
PY
