---
aspect: validation-benchmarks
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-18T11:00:00 CEST
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

Warp WCSPH Euler-step experiment:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-wcsph-euler-step/experiment.md`
- Correctness wrapper: `run_correctness.sh`.
- Focused correctness checks Warp Euler stepping directly and the chained
  `wc_sph_euler_step()` path against CPU reference density, pressure,
  pressure-gradient acceleration, and final position/velocity state.
- The focused Warp SPH/NNPS suite passes with `26 passed`.
- This is a one-step correctness milestone. Repeated-step benchmarking should
  wait for a device-aware NNPS refresh after positions move.

Warp WCSPH leapfrog checkpoint:

- The Euler-step experiment now also records a KDK leapfrog checkpoint.
- Focused tests cover device-coordinate NNPS refresh, direct leapfrog
  kick/drift with periodic position wrapping, and `wc_sph_leapfrog_step()`
  against CPU reference calculations.
- Current focused result:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
29 passed, 2 warnings in 5.82s
```

Warp artificial-viscosity checkpoint:

- `pysph/base/warp_sph.py` now has an additive Monaghan-style artificial
  viscosity kernel for the WCSPH momentum path. It now uses pair-averaged
  per-particle sound speed `cs` when available, with constant `c0` as a
  compatibility fallback.
- Focused tests compare the artificial-viscosity acceleration against a CPU
  CubicSpline reference and verify that the viscosity term adds onto existing
  acceleration arrays instead of replacing them.
- Current focused result:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
32 passed, 2 warnings in 4.14s
```

Warp Tait EOS checkpoint:

- `compute_tait_eos()` matches PySPH `TaitEOS` for pressure and per-particle
  sound speed `cs`.
- Focused tests cover direct Tait EOS output and a small WCSPH Euler step using
  Tait pressure plus `cs`-based artificial viscosity.

Warp elliptical-drop runner:

- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`
  contains the first application-style runner around the current Warp prototype.
- It creates the standard circular elliptical-drop initial patch and velocity
  field, advances with `UniformGridWarpNNPS` and `wc_sph_leapfrog_step()`, pulls
  final arrays once, and writes scalar metrics plus an `.npz` result.
- Smoke result with `nx=8`, `steps=2`, `dt=1.0e-5`, `c0=20.0`, `alpha=0.1`,
  `beta=0.0`, `eos=tait`, `gamma=7.0`: 204 particles, `all_finite=true`,
  final time `2e-05`, `rho_min=0.5834615230560303`,
  `rho_max=0.9999511241912842`, `cs_min=3.9725253582000732`,
  `cs_max=19.997066497802734`, kinetic energy `8078.17389338273`.
- Ramp results stayed finite through `nx=24`, 1808 particles, 10 steps at
  `dt=5.0e-6`, with `rho_min=0.633074939250946`,
  `rho_max=0.9999793767929077`, and kinetic energy `7840.533230601928`.
- Artificial-viscosity ramp check with `nx=16`, 805 particles, 5 steps,
  `dt=1.0e-5`, `alpha=0.1`, `beta=0.0` stayed finite with
  `rho_min=0.6330116391181946`, `rho_max=0.9999754428863525`, and kinetic
  energy `7868.737673401772`.
- Tait EOS ramp check with `nx=16`, 805 particles, 5 steps, `dt=1.0e-5`,
  `alpha=0.1`, `beta=0.0`, `gamma=7.0` stayed finite with
  `rho_min=0.6329819560050964`, `rho_max=0.9999754428863525`,
  `cs_min=5.072288990020752`, `cs_max=19.99852752685547`, and kinetic energy
  `7868.739071212255`.
- XSPH/Gaussian/adaptive-dt checkpoint:
  `pysph/base/warp_sph.py` now supports Gaussian kernel selection,
  `compute_xsph_correction()`, `leapfrog_drift_xsph()`, and
  `compute_wcsph_adaptive_timestep()`. Adaptive dt computes `dt_cfl` and
  `dt_force` on device, reduces them on device, and pulls only the final scalar
  timestep per step.
- Focused Warp SPH result after this checkpoint:

```text
python -m pytest pysph/base/tests/test_warp_sph.py -q
18 passed, 2 warnings in 3.50s
```

- Updated smoke wrapper now exercises Gaussian + Tait + artificial viscosity +
  XSPH + adaptive dt. Result with `nx=8`, 204 particles, 2 steps:
  `all_finite=true`, `kernel=gaussian`, `radius_scale=3.0`,
  `xsph_eps=0.5`, `adaptive_dt=true`, `rho_min=0.534595251083374`,
  `rho_max=0.9998562335968018`, `dt_min_used=9.999999747378752e-06`,
  `dt_max_used=9.999999747378752e-06`, kinetic energy
  `8078.179766857993`.
- New comparison script
  `compare_warp_pysph_elliptical_drop.py` runs the Warp path and a CPU
  PySPH-primitive baseline using `LinkedListNNPS`, `Gaussian`, Tait EOS,
  artificial viscosity, XSPH, and adaptive dt formulas. It writes CPU/Warp
  `.npz` outputs plus `comparison-smoke.png` with side-by-side speed-colored
  scatter plots. Smoke comparison result: CPU and Warp both finite with 204
  particles; CPU `rho_min=0.534595094929311`, Warp
  `rho_min=0.534595251083374`; CPU kinetic energy `8078.179846214378`, Warp
  `8078.179766857993`.
- Resolved `nx=100` Application-backed comparison:
  `resolved_elliptical_drop_comparison.py` runs PySPH's
  `elliptical_drop_no_scheme.py` Application baseline and the Warp runner at
  `t=0.0008` and `t=0.0038`, writes side-by-side images with exact ellipse
  overlays, and records timing/shape/density/energy metrics.
- The first resolved run used Warp summation density while PySPH evolved
  density through `ContinuityEquation`/`WCSPHStep`; it is now diagnostic only.
  That mismatch caused larger density/pressure excursions and many small Warp
  adaptive substeps: Warp took 4807 steps through `t=0.0038`, with density at
  `t=0.0008` ranging from `0.9505811929702759` to `1.0439165830612183`.
- The continuity-density parity run uses the new Warp PEC-style density path
  (`density_mode='continuity'`). At `nx=100`, 31417 particles, PySPH CPU took
  `228.25765374601178` s / 1393 steps, and Warp took
  `30.008050591000938` s / 1804 steps, for `7.606547218180963x` wall-time
  speedup. At `t=0.0038`, major-axis delta was
  `1.5947661098358878e-06`, minor-axis delta was
  `2.1943316564909665e-06`, `rho_min` delta was
  `-4.816405699936688e-06`, `rho_max` delta was
  `8.755722542552746e-07`, and kinetic-energy delta was
  `-0.00043376772100600647`.
- The old continuity-density run's remaining step-count difference was caused
  by timestep policy, not the density formulation: the Warp runner capped
  adaptive `dt` to the initial value while PySPH allowed the damped timestep to
  grow after the `n_damp` ramp.
- The timestep-policy parity run adds a PySPH-like policy to the resolved
  runner: damp early timesteps with `n_damp`, allow adaptive growth through an
  undamped `warp_dt_max`, and apply checkpoint landing caps only to the current
  step. At `nx=100`, 31417 particles, PySPH CPU took
  `233.97314716299297` s / 1393 steps, and Warp took
  `23.629107111992198` s / 1393 steps, for
  `9.901903870258701x` wall-time speedup. At `t=0.0038`, major-axis delta was
  `-3.1258252297661215e-07`, minor-axis delta was
  `1.3598666296354978e-06`, `rho_min` delta was
  `-8.228944999855159e-07`, `rho_max` delta was
  `9.947815438060559e-07`, and kinetic-energy delta was
  `-0.0004318240680731833`. The very small Warp `dt_min` is an output-time
  landing step and does not permanently shrink later adaptive steps.
- Precision note for the resolved Warp runner: initial host arrays are created
  as `float64`, but `WarpDeviceHelper` casts floating properties to
  `compyle.config.get_config().use_double`. On the active machine this config
  is `False`, so the current Warp device execution path is fp32. An explicit
  Warp-only `nx=100` rerun under that fp32 config took
  `26.5732471299998` s / 1393 steps, versus the committed run's
  `23.629107111992198` s / 1393 steps; both are the same fp32 device path, so
  the timing difference is treated as run-to-run/module-cache variance.
- Million-particle fixed-step comparison before cache reuse:
  `nx=565`, 1,002,885 particles, 10 fixed steps. PySPH CPU Application took
  57.48 s, Warp GPU took 7.17 s, for `8.01673640167364x` wall-time speedup.
  Final shape/density/energy deltas were tiny and both outputs were finite.
- Profiling that million-particle path showed the continuity-density PEC step
  was building eight same-array neighbor caches per step. Each cache had about
  45M neighbor entries, and cache construction dominated the Warp step time.
- Cache-reuse slice: `pysph/base/warp_sph.py` helpers now accept optional
  prebuilt neighbor caches, and the continuity-density PEC path builds one
  cache per half-stage. Focused tests assert the full step builds two equation
  caches instead of eight.
- Million-particle fixed-step comparison after cache reuse:
  the same PySPH CPU Application baseline remains 57.48 s, while Warp improved
  to 4.51 s / 10 steps, for `12.7450110864745x` speedup and
  `1.58980044345898x` improvement over the prior Warp run. The segmented
  cache profile now shows two cache builds per step and step wall times around
  `0.098335` to `0.138290` s before runner output/setup overhead.
- Equation-fusion slice (ADR-0003 dynamic code generation): the four
  continuity-stage neighbor-loop equations are fused into one generated kernel
  per PEC half-stage. The segmented million-particle profile drops equation
  launches from 8 to 2 per step, equation-kernel time from ~0.064-0.088 s to
  0.011-0.014 s per step (~5-6x), and steady-state step wall from
  0.098-0.138 s to 0.076-0.098 s (~25%); the neighbor-cache build (~0.034-0.046
  s) is now the dominant per-step cost. The 10-step headline wall is
  overhead/IO-bound and noisy (warm samples 3.77-6.04 s, best 3.77 s =
  `15.25x` vs CPU), so per-step compute is the meaningful metric. Numerical
  parity is essentially exact versus the prior separate-kernel Warp run
  (positions/density/pressure identical to fp32 print precision, kinetic-energy
  delta `-6.4e-09`); CPU deltas match the cache-reuse run. The new focused suite
  is `python -m pytest -q pysph/base/tests/test_warp_codegen.py
  pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
  -> `47 passed`, including a generated-vs-hand-helper parity test and a
  single-fused-launch-per-stage count test.
- Adaptive `nx=100` resolved guard with the fused path: reached `t=0.0038` in
  `1393` steps (identical to the committed run), all finite, with shape deltas
  `~4.8e-07`, density deltas `~1e-06`, and kinetic-energy delta `1.45e-04`
  versus committed Warp metrics. The adaptive path now runs
  fused(1)+dt_factors(1)=2 traversals per stage instead of 5, and its Warp wall
  fell from the committed `23.63 s` to `10.83-14.33 s` (cross-session, same step
  count) -- a cleaner demonstration of the fusion because that run is
  per-step-compute-bound.
- Grid-direct neighbor traversal slice (ADR-0004): the continuity hot path
  stops materializing a flat CSR neighbor list; both consumers walk the cell
  list directly. Segmented million-particle (`nx=565`) profile via
  `profile_grid_direct_neighbors.py`: `build_neighbor_cache_gpu` called 0 times
  on the continuity path; the dominant flat-cache-build term (~0.034-0.046
  s/step) is gone; grid build now ~0.0004-0.0007 s; equation kernel rises
  0.011-0.014 -> 0.023-0.025 s/launch (it absorbs the single cutoff traversal);
  step wall (steady) 0.076-0.098 -> 0.059-0.064 s (~25-35% lower); KE delta vs
  flat fused `-1.99e-06`, all finite. Adaptive `nx=100` resolved guard kept
  exactly `1393` steps with fp32-scale deltas (KE relative `1.5e-08`, shape
  `~2.4e-7/6.6e-7`, density `~9e-7`) and Warp wall `7.82 s`. Focused suite
  `50 passed` (adds grid-vs-flat fused parity, grid cache-distinct + single-cell
  numeric parity, and a zero-flat-cache-build assertion). Summary folder
  `million-cpu-gpu-grid-direct/`.

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
- One-step WCSPH chain correctness.
- Device-aware repeated-step refresh criteria.
- KDK leapfrog correctness.
- Periodic position wrapping correctness.
- Application-style Warp elliptical-drop smoke metrics.
- Artificial-viscosity acceleration correctness and smoke metrics.
- Tait EOS and per-particle sound-speed correctness and smoke metrics.
- Gaussian kernel correctness.
- XSPH leapfrog correction correctness.
- Device-reduced adaptive timestep correctness.
- CPU PySPH-primitive side-by-side image comparison.
- Optional parallel/Zoltan test slice after commit readiness.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: validation-benchmarks` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `gpu-nnps`, `particle-memory`, and `warp-backend`.
- Influences: success criteria and review evidence.
