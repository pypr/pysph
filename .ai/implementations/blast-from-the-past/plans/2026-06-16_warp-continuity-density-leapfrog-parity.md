---
type: plan
id: 2026-06-16_warp-continuity-density-leapfrog-parity
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T23:58:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp Continuity-Density Leapfrog Parity

## Goal

Make the resolved elliptical-drop comparison apples to apples by adding a Warp
step path that evolves density with `ContinuityEquation` and the WCSPH
integrator density stages, matching the default PySPH no-scheme CPU
Application path.

The existing resolved `nx=100` output is diagnostic only: it showed Warp is
faster per step, but it also exposed that the Warp runner was recomputing
density with summation density while PySPH evolves `rho` through `arho`.

## Baseline to match

PySPH `pysph/examples/elliptical_drop_no_scheme.py` uses:

- `TaitEOS` in a non-real group;
- `ContinuityEquation(dest='fluid', sources=['fluid'])`;
- `MomentumEquation(dest='fluid', sources=['fluid'], alpha=0.1, beta=0.0)`;
- `XSPHCorrection(dest='fluid', sources=['fluid'])`;
- `WCSPHStep`, where `rho` is advanced from saved `rho0` with `arho` in the
  integrator stages.

The Warp parity path should therefore compute `arho` on device and integrate
`rho` on device, rather than replacing `rho` each force evaluation with
summation density.

## Approach

### Phase 1 - Device density-stage kernels

- Add small Warp kernels for WCSPH density integration stages:
  - save/restore stage inputs if needed (`rho0` at the beginning of a step);
  - update `rho = rho0 + 0.5*dt*arho` for the midpoint stage;
  - update `rho = rho0 + dt*arho` for the full stage.
- Keep these kernels device-resident and reuse existing ParticleArray
  properties. If temporary storage is required, prefer existing PySPH-style
  properties (`rho0`) when present and create only the minimum helper needed
  for the experiment path.

### Phase 2 - Selectable WCSPH force density mode

- Extend the internal Warp WCSPH acceleration helper so density preparation is
  selectable:
  - `density_mode='summation'` keeps existing behavior and tests stable;
  - `density_mode='continuity'` assumes current `rho` is authoritative,
    computes EOS from it, computes momentum, and computes `arho` with
    `compute_continuity()`.
- Preserve no-unnecessary-transfer behavior:
  - no full host particle pulls during repeated stepping;
  - no full host pushes after the initial setup;
  - adaptive timestep continues to pull only the reduced scalar `dt`.

### Phase 3 - Continuity leapfrog step

- Add continuity-density support to `wc_sph_leapfrog_step()` through a
  conservative keyword, for example `density_mode='summation'`.
- For `density_mode='continuity'`, mirror the PySPH `WCSPHStep` staging as
  closely as the current KDK-style helper permits:
  - compute force/EOS/continuity at the current state;
  - select adaptive `dt`;
  - advance velocity and density to the intermediate state;
  - drift positions with velocity/XSPH correction;
  - rebuild NNPS from device positions;
  - recompute force/EOS/continuity;
  - advance velocity and density to the full step.
- Keep summation-density behavior as the default until the continuity path is
  validated.

### Phase 4 - Tests

- Add focused tests proving the new density-stage kernels match CPU reference
  arithmetic.
- Add a focused `wc_sph_leapfrog_step(..., density_mode='continuity')` test
  against a hand-computed CPU reference using `ContinuityEquation` and the same
  Tait/EOS + pressure-gradient chain.
- Preserve the existing summation-density leapfrog test.

### Phase 5 - Rerun resolved comparison

- Update the Warp elliptical-drop runner and resolved comparison script to use
  `density_mode='continuity'` for the PySPH Application comparison.
- Rerun the default `nx=100` resolved comparison at:
  - `t = 0.0008`;
  - `t = 0.0038`.
- Update the experiment report to clearly separate:
  - the previous summation-density diagnostic run;
  - the new continuity-density parity run.

## Files expected to change

- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_sph.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- New/updated resolved comparison outputs under the same experiment directory.
- Memory/context:
  - `.ai/implementations/blast-from-the-past/current.md`
  - `.ai/implementations/blast-from-the-past/aspects/validation-benchmarks/context.md`
  - `.ai/implementations/blast-from-the-past/aspects/gpu-nnps/context.md`
  - `.ai/implementations/blast-from-the-past/aspects/particle-memory/context.md`
  - `.ai/implementations/blast-from-the-past/updates/daily/2026-06-16.md`
  - session log for this slice

No boundary amendment is expected because `pysph/base/warp_sph.py` and
`pysph/base/tests/test_warp_sph.py` are already inside the approved prototype
boundary.

## Tests / validation

- Focused test suite:
  `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- Resolved comparison script:
  `python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-continuity --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved --max-steps 10000000`
- Confirm CPU and Warp both report finite metrics at each checkpoint.
- Confirm side-by-side images exist and include exact ellipse overlays.
- Confirm the continuity run no longer uses summation density during repeated
  Warp stepping.
- Confirm no full-array host/device transfers occur inside the repeated Warp
  stepping loop beyond the scalar adaptive timestep handoff.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai/implementations/blast-from-the-past pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py`

## Success criteria

- Existing summation-density leapfrog behavior remains tested and unchanged by
  default.
- New continuity-density leapfrog behavior passes focused CPU-reference tests.
- The resolved `nx=100` comparison runs through `t=0.0008` and `t=0.0038`
  using `density_mode='continuity'` on Warp.
- The reported CPU/Warp step counts, timestep ranges, shape metrics, density
  ranges, kinetic energy, and wall times are updated from the continuity run.
- Any remaining difference is described as a numerical/integrator parity issue,
  not hidden behind a formulation mismatch.

## Risks

- The current Warp helper is KDK-shaped while PySPH `WCSPHStep` is predictor
  corrector/PEC-style. Matching density evolution may expose a remaining
  integrator-stage mismatch even after replacing summation density.
- If `rho0` storage is absent from the ad-hoc Warp ParticleArray, the runner
  may need to add the property during setup.
- Adaptive timestep behavior may still differ if the force estimate uses a
  different acceleration stage than PySPH's solver.
- The resolved `nx=100` rerun may take several minutes again.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T23:59:00 CEST
- Approval, verbatim quote:
  > APPROVED
