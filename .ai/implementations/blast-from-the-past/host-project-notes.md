# Host Project Notes

Just enough about PySPH for `blast-from-the-past` to integrate cleanly. This is not a host catalogue.

## Stack and Toolchain

- Language/runtime: Python plus Cython extension modules and C++-mode Cython for several low-level paths.
- Build command: `python setup.py build_ext --inplace` or `make build`.
- Default test command: `python -m pytest -m "not slow" pysph` or `make test`.
- Full test command: `python -m pytest pysph` or `make testall`.
- Parallel/Zoltan test command from CI: `python -m pytest -v -m 'slow or parallel'`.
- Lint/format: no explicit root `ruff`, `black`, `isort`, `flake8`, or `mypy` config found during scaffold discovery. Confirm with team before introducing new style tooling.

## Integration Boundary

- `pysph/**/*.pxd` - public Cython declarations and extension ABI surface.
- `pysph/**/*.pyx` - implementation files for Cython particle arrays, NNPS, kernels, MPI exchange, and mesh tooling.
- `pysph/base/gpu_nnps.py` - re-export module for GPU NNPS classes.

## Boundary Surface Observed During Discovery

- `pysph/base/gpu_nnps_base.pxd` declares `GPUNeighborCache`, `GPUNNPS`, and `BruteForceNNPS`.
- `pysph/base/gpu_nnps_base.pyx` implements GPU neighbor cache allocation, GPU-to-CPU neighbor retrieval, bounds computation, and brute-force PyOpenCL neighbor kernels.
- `pysph/base/gpu_nnps.py` re-exports `GPUNeighborCache`, `GPUNNPS`, `BruteForceNNPS`, `ZOrderGPUNNPS`, `StratifiedSFCGPUNNPS`, `GPUDomainManager`, and `OctreeGPUNNPS`.
- Broader `.pxd/.pyx` surface includes `ParticleArray`, `DomainManager`, `NNPS`, CPU/GPU NNPS variants, kernels, point/linalg helpers, `ParallelManager`, and mesh tools.

## Host Conventions We Inherit

- Pytest default excludes `slow` tests via `setup.cfg` and `tox.ini`.
- Build/test commands should follow the existing Makefile and CI conventions unless an ADR approves a change.
- Cython files use `# cython: language_level=3, embedsignature=True` in many active modules. Confirm with team before changing Cython compiler directives.

## Pre-Existing Host AI Configs

- None found during scaffold discovery among `.cursorrules`, `AGENTS.md`, `CLAUDE.md`, `OPENAI.md`, `.aider.conf.yml`, and `.github/copilot-instructions.md`.

## Secrets Locations

- (none identified; do not copy secrets into `.ai/`)

## Out-of-Scope Zones

- Host application code outside `pysph/**/*.pxd`, `pysph/**/*.pyx`, and the discovered GPU NNPS export file unless a boundary amendment is approved.
- General SPH formulation redesign.
