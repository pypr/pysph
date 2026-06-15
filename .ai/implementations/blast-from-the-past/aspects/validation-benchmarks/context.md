---
aspect: validation-benchmarks
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T09:30:00 CET
status: active
---

# Aspect: validation-benchmarks

## What this aspect covers

Baselines, timings, correctness checks, acceptance thresholds, experiment handoff, and validation evidence for fast particle dynamics.

## Current understanding

The first correctness baseline is ParticleArray and DeviceHelper behavior parity: construction, scalar broadcast, strided properties, constants, push/pull, alignment, add/remove/extract/append, serialization metadata, and GPU ordering allowances. Performance criteria are now split into correctness gates for the current prototype and timing evidence for the next optimization step.

The active Python can import Warp `1.14.0`. The venv initially lacked `compyle`, `cyarray`, and PySPH's compiled `pysph.base.particle_array` extension; installing the declared requirements and rebuilding `particle_array` narrowly with `pyximport` made the focused tests runnable. Isolated `WarpArray.aligned()` probes passed on `cuda:0` for float64 strided data and int64 tag data.

Later host-integration work installed PySPH editable into the PQT venv with PyZoltan/Zoltan enabled, so tests now run against the installed editable package instead of only the narrow `pyximport` build.

Current passing checks:

- `python -m pytest -q pysph/base/tests/test_warp_device_helper.py` - 20 passed.
- Python-launched CPU sanity slice for constructor, alignment, add-property, constants, remove, add, and extract - 7 passed.
- Plain import validation for PySPH/PyZoltan/Zoltan/parallel manager - pass; `has_zoltan()` and `in_parallel()` are `True`.

Active experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_initial-warp-benchmark-placeholder/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_mutation_benchmark.sh`.
- Smoke result: `results-smoke-20260615.txt`; Warp add/remove/extract are currently slower than CPU because the prototype still uses host-side rebuilds/readback for structural mutations.

Next benchmark family should target NNPS:

- CPU-vs-Warp neighbor set correctness.
- NNPS update time.
- all-particle query time.
- cache build time.
- readback time separated from device computation.
- average neighbor count and smoothing-length mode recorded with each run.

## Key sub-topics

- Baseline selection.
- Hardware/runtime recording.
- Correctness tolerance and performance thresholds.
- ParticleArray/DeviceHelper parity suite.
- Performance benchmarks for structural mutations and device sync.
- NNPS benchmark fixtures and timing thresholds.
- Optional parallel/Zoltan test slice after commit readiness.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: validation-benchmarks` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `gpu-nnps`, `particle-memory`, and `warp-backend`.
- Influences: success criteria and review evidence.
