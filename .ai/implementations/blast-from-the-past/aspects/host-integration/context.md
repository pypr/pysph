---
aspect: host-integration
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T09:45:00 CET
status: active
---

# Aspect: host-integration

## What this aspect covers

CLI/build/test integration, compatibility with existing OpenCL/CUDA/Compyle paths, and keeping the implementation boundary truthful.

## Current understanding

PySPH is installed editable in the active PQT venv at `/home/kunalp/.pqt_venv_e0b41259`. Zoltan `v3.901` was built from `sandialabs/Zoltan` under `/home/kunalp/prediqt/zoltan`, then PyZoltan `1.1.1` was installed with `ZOLTAN=/home/kunalp/prediqt/zoltan` and `--no-build-isolation`.

Persistent rebuild configuration lives in `/home/kunalp/.compyle/config.py`, with `ZOLTAN='/home/kunalp/prediqt/zoltan'` and MPI flags from the PQT OpenMPI Spack view. Setuptools was installed into the venv so Python 3.14 can import `distutils` through `setuptools._distutils`.

Validation showed plain imports work for `pysph`, `pyzoltan`, `pysph.parallel.parallel_manager`, and the Warp ParticleArray path; `has_mpi()`, `has_zoltan()`, and `in_parallel()` all return `True`.

The top-level `CODEBASE_UNDERSTANDING.md` has been worked into
`.ai/implementations/blast-from-the-past/spec/codebase-understanding/` as a
curated implementation-facing digest. The spec now has an index at
`.ai/implementations/blast-from-the-past/spec/README.md` linking the codebase,
ParticleArray, and NNPS layers.

## Key sub-topics

- Existing build/test commands.
- Optional GPU dependencies.
- Boundary amendments and review integrity.
- Output/restart/dummy-particle compatibility.
- Local PQT editable install and Zoltan/PyZoltan rebuild reproducibility.
- Codebase-level architecture context for staged GPU migration.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: host-integration` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `cython-boundary` - approved host files.
- Influences: all implementation plans and reviews.
