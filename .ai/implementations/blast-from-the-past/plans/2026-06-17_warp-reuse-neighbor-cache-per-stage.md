---
type: plan
id: 2026-06-17_warp-reuse-neighbor-cache-per-stage
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-17T16:55:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, validation-benchmarks]
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp Reuse Neighbor Cache Per Stage

## Goal

Redo the million-particle headline benchmark after reducing the Warp
continuity-density WCSPH step from eight neighbor-cache builds per step toward
two cache builds per step.

The current fixed-step `nx=565` benchmark shows:

- 1,002,885 particles;
- PySPH CPU Application: 57.48 s / 10 steps;
- Warp GPU: 7.17 s / 10 steps;
- speedup: 8.01673640167364x.

The segmentation run showed the main bottleneck: each full Warp step builds the
same same-array neighbor cache eight times. Each cache contains about 45M
neighbor entries, and cache construction alone is roughly 70% of the GPU step
time.

## Context

In the continuity-density PEC path, each half-stage currently calls:

- pressure gradient;
- artificial viscosity;
- continuity;
- XSPH;

and each helper calls `nnps.build_neighbor_cache_gpu()` internally. The NNPS
is only updated after the half-stage drift and final stage, so all four
neighbor-loop equations within a stage can safely consume the same cache.

## Approach

1. Extend the existing Warp equation helpers with an optional `cache=None`
   parameter:
   - `compute_summation_density`;
   - `compute_continuity`;
   - `compute_pressure_gradient`;
   - `compute_artificial_viscosity`;
   - `compute_xsph_correction`;
   - `compute_wcsph_adaptive_timestep`, if useful for the adaptive path.
2. Preserve the public/default behavior: when `cache is None`, helpers build
   the cache internally as they do today.
3. In `_wc_sph_pec_continuity_step`, build one cache before each half-stage's
   neighbor-loop equations, then pass it into acceleration/continuity/XSPH
   helpers for that stage.
4. Keep NNPS rebuilds after coordinate updates exactly where they are now:
   after the half-stage and final stage.
5. Add focused tests that monkeypatch/count `build_neighbor_cache_gpu()`:
   - direct helper calls still build one cache by default;
   - a continuity-density `wc_sph_leapfrog_step()` with viscosity and XSPH
     builds two equation caches per step instead of eight.
6. Rerun the bounded million-particle headline benchmark:
   - CPU fixed-step Application baseline can reuse the recorded 57.48 s if no
     CPU inputs changed, but rerun if needed for a clean same-session number;
   - Warp fixed-step `nx=565`, 10 steps, same physics/settings;
   - record compute and wall-time numbers.

## Files expected to change

- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_sph.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/` for a compact new summary JSON
- `.ai/implementations/blast-from-the-past/current.md`
- `.ai/implementations/blast-from-the-past/updates/daily/2026-06-17.md`
- session log for this slice
- review artifact before commit

## Tests / validation

- `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- Focused cache-count test for the continuity-density step.
- Segmented one-step `nx=565` timing to verify two cache builds per step.
- Fixed-step million-particle benchmark:

```text
python pysph/examples/elliptical_drop_no_scheme.py --nx 565 --tf 0.000003732778967800475 --timestep 0.0000003732778967800475 --no-adaptive-timestep --n-damp 0 --pfreq 10 --fname million-pysph --directory ... --logfile '' --quiet
python .../warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output ...
```

- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past`

## Success criteria

- The full continuity-density step with artificial viscosity and XSPH builds
  two equation neighbor caches per step, not eight.
- Focused tests still pass.
- The million-particle fixed-step Warp timing improves materially from the
  previous 7.17 s / 10 steps.
- CPU/GPU final-state deltas remain near the previous small values.
- No full-duration multi-hour run is launched.

## Risks

- Passing a stale cache across a coordinate update would be incorrect. The
  implementation must only reuse the cache within a stage before drift/stage
  coordinate updates.
- Reusing one cache also reuses the flat neighbor allocation for several
  kernels; this should be correct, but tests need to cover numerical parity.
- We may reveal a second bottleneck after cache builds are reduced, such as
  separate equation-kernel traversals or synchronization after every helper.

## Out of scope

- Fusing pgrad, viscosity, continuity, and XSPH into a single kernel.
- Removing all synchronizations.
- Full `nx=565`, `tf=0.0076` GPU-only elliptical-drop run.
- MPI/OpenMP CPU benchmark changes.

## Estimated effort

One focused implementation/benchmarking session.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-17T17:13:22 CEST
- Approval, verbatim quote:
  > approved
