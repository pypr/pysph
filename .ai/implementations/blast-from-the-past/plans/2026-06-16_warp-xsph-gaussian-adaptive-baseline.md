---
type: plan
id: 2026-06-16_warp-xsph-gaussian-adaptive-baseline
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T17:35:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
within_boundary: false
---

# Plan: Warp XSPH, Gaussian Kernel, Adaptive DT, and PySPH Comparison

## Goal

Move the Warp elliptical-drop runner from a finite smoke workload toward an
apples-to-apples validation case by adding:

1. optional XSPH correction on the GPU;
2. Gaussian kernel support on the GPU;
3. adaptive timestep metrics/reduction on the GPU;
4. CPU PySPH baseline comparison with side-by-side images.

The runner must avoid unnecessary host/device transfers: no full particle-array
pulls during stepping. Transfers are limited to necessary scalar adaptive
timestep values and explicit checkpoint/output arrays for validation and plots.

## Context

Current committed state has:

- `UniformGridWarpNNPS.update(push=False)` for device-authoritative refresh.
- KDK leapfrog drift/kick on Warp.
- CubicSpline density, pressure-gradient, artificial viscosity, Tait EOS, and
  per-particle `cs`.
- A standalone Warp elliptical-drop runner that writes metrics and `.npz`
  outputs after a run.

PySPH references:

- `pysph.sph.basic_equations.XSPHCorrectionForLeapFrog` computes only the XSPH
  correction term:

```text
ax_i = -eps * sum_j m_j * (u_i - u_j) * W_ij * 2/(rho_i + rho_j)
```

- `LeapFrogStep` uses the correction during position updates as
  `x += dt * (u + ax)`.
- `pysph.base.kernels.Gaussian` has radius scale `3.0` and:

```text
W(q) = sigma_g * exp(-q*q), q < 3
dW/dq = -2*q*exp(-q*q)*sigma_g, q < 3
```

- WCSPH adaptive-timestep terms include `dt_cfl`, `dt_force`, and integrator
  reductions over those per-particle fields.

## Approach

### Phase 1 - XSPH on GPU

- Add Warp kernels for XSPH correction writing `ax`, `ay`, `az`.
- Add `compute_xsph_correction(nnps, eps=0.5, ...)`.
- Add a drift variant that uses `u + ax`, `v + ay`, `w + az`.
- Wire optional `xsph_eps=None` or `0.0` into `wc_sph_leapfrog_step()`.
- For KDK, compute XSPH after the first half-kick and before drift, so the
  correction uses half-step velocities.
- Add focused CPU-reference tests.

### Phase 2 - Gaussian Kernel on GPU

- Add Gaussian `W` and `dW/dq` Warp functions for float32/float64.
- Make equation kernels select `kernel='cubic'` or `kernel='gaussian'`.
- Ensure the runner and NNPS use `radius_scale=3.0` for Gaussian.
- Add focused tests comparing Gaussian density and pressure-gradient values
  against PySPH `Gaussian(dim=2)`.
- Switch the elliptical-drop runner default to Gaussian.

### Phase 3 - Adaptive DT on GPU

- Add per-particle device fields as needed: `dt_cfl`, `dt_force`, and any
  minimal temporary fields.
- Update momentum/adaptive kernels to compute:
  - CFL factor comparable to PySPH WCSPH momentum;
  - force factor from acceleration magnitude;
  - min/selected timestep via GPU reductions.
- In the Python runner, pull only the final scalar timestep per step. This is a
  necessary transfer while stepping is orchestrated from Python and launch
  parameters are host scalars.
- Add `--adaptive-dt`, `--cfl`, `--n-damp`, `--dt-min`, and `--dt-max` style
  runner controls, following PySPH semantics where practical.
- Add tests for device-computed timestep factors and a short adaptive smoke.

### Phase 4 - CPU PySPH baseline and side-by-side images

- Add an experiment comparison script under the elliptical-drop experiment
  directory.
- Run a CPU PySPH baseline with the matched formulation:
  Gaussian + Tait EOS + artificial viscosity + optional XSPH + adaptive dt.
- Capture checkpoints at PySPH example output times where practical
  (`0.0008`, `0.0038`) or document a shorter smoke comparison if runtime is too
  high.
- Generate side-by-side images from CPU and Warp checkpoints, likely scatter
  plots colored by speed or density plus optional outline/axis-equal panels.
- Record metrics: particle count, time, dt history summary, bounds, density
  range, kinetic energy, and image paths.

## Files expected to change

- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_sph.py`
- `pysph/base/warp_nnps.py` if Gaussian radius-scale handling needs NNPS
  support beyond runner configuration.
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- New comparison/plot scripts under
  `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- `.ai/implementations/blast-from-the-past/current.md`
- `.ai/implementations/blast-from-the-past/aspects/gpu-nnps/context.md`
- `.ai/implementations/blast-from-the-past/aspects/validation-benchmarks/context.md`
- `.ai/implementations/blast-from-the-past/updates/daily/2026-06-16.md`
- session log for this slice

## Boundary note

These files are part of the approved Python Warp prototype surface, but the
memory validator treats `pysph/base/warp_*.py` literally rather than as a glob.
This plan is marked `within_boundary: false` and should call that out again in
review.

## Tests / validation

- `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- `bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- Short adaptive Gaussian+XSPH GPU smoke with no full particle pulls inside the
  step loop.
- CPU PySPH baseline smoke/comparison script.
- Side-by-side image generation check.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past`

## Risks / sequencing

- This is a larger slice than the prior equation kernels. If it gets too large,
  stop after a defined checkpoint in this order: XSPH, Gaussian, adaptive dt,
  baseline/images.
- Adaptive timestep may require one scalar host transfer per step; that is
  necessary with the current Python launch loop and should be documented.
- Full PySPH baseline at example output times may be slower than the smoke
  budget. If so, run a smaller/narrower baseline first and record it honestly.
- Gaussian radius scale changes neighbor counts, so the runner must construct
  `UniformGridWarpNNPS(radius_scale=3.0)` for Gaussian.

## Production-readiness expectation

Completing this plan should make the runner much closer to a production
elliptical-drop validation path. It still may not be production-ready until the
side-by-side baseline comparison has accepted tolerances and the team decides
whether standalone runner output is sufficient or PySPH `Application/Solver`
integration is required.

## Estimated effort

One long implementation session, likely with checkpointing after each phase.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T22:27:35 CEST
- Approval, verbatim quote:
  > APPROVED
