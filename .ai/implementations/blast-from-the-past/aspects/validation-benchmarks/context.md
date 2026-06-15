---
aspect: validation-benchmarks
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T14:25:00 CET
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
- Smoke result is recorded in the ParticleArray experiment doc; Warp add/remove/extract are currently slower than CPU because the prototype still uses host-side rebuilds/readback for structural mutations.

Next benchmark family should target NNPS:

- CPU-vs-Warp neighbor set correctness.
- NNPS update time.
- all-particle query time.
- cache build time.
- readback time separated from device computation.
- average neighbor count and smoothing-length mode recorded with each run.

First NNPS experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_benchmark.sh`.
- Smoke result is recorded in the experiment doc; CPU, uncached Warp, cached
  Warp, and Warp grid average neighbor counts match at 128 particles. Cached
  Warp brute force is much faster than the per-query path but remains an O(N^2)
  bridge; Warp grid is the first cell-list baseline.
- The smoke benchmark now records CPU/GPU hardware and CPU-relative speedup. On
  Intel(R) Core(TM) Ultra 7 155H versus NVIDIA GeForce RTX 4060 Laptop GPU, the
  128-particle smoke run shows `warp_grid` at `0.041x` CPU speed.
- A 1,000,000-particle host-facing benchmark on the same hardware shows
  `warp_grid` at `4.269x` CPU speed with matching average neighbor count
  (`25.568`).
- A 1,000,000-particle device-oriented benchmark shows `warp_grid_device` at
  `88.288x` CPU speed with matching average neighbor count (`25.568`). This is
  the relevant GPU-side result because it avoids the per-particle
  `get_nearest_particles()`/`UIntArray` loop.

Device-consumption NNPS experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-device-consumption/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_benchmark.sh`.
- Focused Warp NNPS correctness now includes direct packed-GPU-cache index
  parity tests and device-side neighbor-sum tests, and passes with `17 passed`.
- A 1,000,000-particle benchmark on Intel(R) Core(TM) Ultra 7 155H versus
  NVIDIA GeForce RTX 4060 Laptop GPU shows `warp_grid_reduce` at `145.583x`
  CPU speed for a neighbor mass sum. Average neighbor sum matches to reported
  precision (`25.568`), with aggregate checksum delta `6` over roughly `25.6M`
  contributions.

Warp summation-density experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-summation-density/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_benchmark.sh`.
- Focused correctness checks `compute_summation_density()` against a CPU
  `CubicSpline` reference in 2D and cross-array 3D and passes with
  `19 passed` across Warp SPH and NNPS tests.
- The CPU baseline is PySPH `SPHEvaluator` with Cython backend,
  `SummationDensity`, `CubicSpline(dim=2)`, and `LinkedListNNPS`.
- A 1M-to-10M sweep on Intel(R) Core(TM) Ultra 7 155H versus NVIDIA GeForce RTX
  4060 Laptop GPU shows matching checksums to reported precision and speedups:
  `152.508x` at 1M, `227.555x` at 2M, `64.964x` at 5M, and `69.084x` at 10M.

Warp EOS+continuity experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-eos-continuity/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_benchmark.sh`.
- Focused correctness checks Warp `IsothermalEOS` and `ContinuityEquation`
  against CPU references and passes with `22 passed` across Warp SPH and NNPS
  tests.
- The CPU baseline is PySPH `SPHEvaluator` with Cython backend,
  `IsothermalEOS`, `ContinuityEquation`, `CubicSpline(dim=2)`, and
  `LinkedListNNPS`.
- The benchmark is capped at 5M particles. A 1M/2M/5M sweep on Intel(R)
  Core(TM) Ultra 7 155H versus NVIDIA GeForce RTX 4060 Laptop GPU shows
  pressure checksums matching to reported precision and speedups: `161.063x`
  at 1M, `136.886x` at 2M, and `72.583x` at 5M.

Warp pressure-gradient experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-pressure-gradient/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Timing wrapper: `run_benchmark.sh`.
- Focused correctness checks Warp inviscid pressure gradient against CPU
  references and passes with `24 passed` across Warp SPH and NNPS tests.
- The CPU baseline is PySPH `SPHEvaluator` with Cython backend, a pure
  `PressureGradientOnly` equation, `CubicSpline(dim=2)`, and `LinkedListNNPS`.
- The benchmark is capped at 5M particles. A 1M/2M/5M sweep on Intel(R)
  Core(TM) Ultra 7 155H versus NVIDIA GeForce RTX 4060 Laptop GPU shows
  speedups: `148.884x` at 1M, `129.854x` at 2M, and `38.722x` at 5M.

## Key sub-topics

- Baseline selection.
- Hardware/runtime recording.
- Correctness tolerance and performance thresholds.
- ParticleArray/DeviceHelper parity suite.
- Performance benchmarks for structural mutations and device sync.
- NNPS benchmark fixtures and timing thresholds.
- Warp grid optimization and device-resident neighbor-list metrics.
- Device-consumption benchmark metrics.
- SPH equation-kernel correctness and operation speedup.
- EOS/continuity capped benchmark metrics.
- Pressure-gradient capped benchmark metrics.
- Optional parallel/Zoltan test slice after commit readiness.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: validation-benchmarks` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `gpu-nnps`, `particle-memory`, and `warp-backend`.
- Influences: success criteria and review evidence.
