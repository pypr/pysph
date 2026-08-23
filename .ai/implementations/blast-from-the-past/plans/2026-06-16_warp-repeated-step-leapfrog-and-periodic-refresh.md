---
type: plan
id: 2026-06-16_warp-repeated-step-leapfrog-and-periodic-refresh
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T11:58:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_nnps.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_nnps.py, pysph/base/tests/test_warp_sph.py]
within_boundary: false
---

# Plan: Warp repeated step leapfrog and periodic refresh

## Goal

Add the first repeated-step Warp dynamics path:

- rebuild `UniformGridWarpNNPS` from device-updated particle coordinates;
- add a minimal leapfrog/KDK-style WCSPH step on the device;
- add periodic position wrapping for drifted coordinates;
- keep host/device transfers out of the inner step except the existing small
  neighbor-length sizing readback;
- validate with focused correctness tests against CPU reference calculations.

## Context

The current one-step `wc_sph_euler_step()` computes density, pressure,
pressure-gradient acceleration, and velocity/position update on the GPU. It is
not safe for repeated GPU stepping yet because `UniformGridWarpNNPS.update()`
unconditionally pushes host `x/y/z/h` back to the device before rebuilding.
That clobbers device-updated positions from the previous step.

The existing equation helpers already expose `push=False` for downstream
stages; the repeated loop should make the device arrays authoritative after the
initial push.

## Approach

1. Extend `UniformGridWarpNNPS.update()` with a narrow option such as
   `push=True`, preserving current behavior for host-side mutation tests while
   allowing repeated GPU loops to call `update(push=False)`.
2. Add Warp kernels/helpers in `pysph/base/warp_sph.py` for:
   - leapfrog half-kick;
   - drift;
   - optional periodic wrap over provided bounds;
   - a convenience `wc_sph_leapfrog_step(...)` that evaluates acceleration,
     half-kicks, drifts/wraps, refreshes NNPS from device data, reevaluates
     acceleration, and completes the half-kick.
3. Keep equation helpers from pushing stale host data in the repeated loop.
   Use existing `push=False` where available and add it only where needed.
4. Add tests:
   - `UniformGridWarpNNPS.update(push=False)` sees device-side position changes
     while default `update()` still supports host mutation;
   - periodic wrap keeps coordinates inside the supplied domain;
   - leapfrog step matches a CPU fixture for one small non-periodic step;
   - repeated stepping does not regress to stale host coordinates.

## Files expected to change

- `pysph/base/warp_nnps.py`
- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_nnps.py`
- `pysph/base/tests/test_warp_sph.py`
- Possibly the active WCSPH experiment doc if validation results or scope need
  to be recorded.

## Tests / validation

- `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- pysph/base/warp_nnps.py pysph/base/warp_sph.py pysph/base/tests/test_warp_nnps.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past`

## Risks

- This plan touches Python host files outside the current integration boundary.
  The review should either amend the boundary or explicitly flag the drift.
- Periodic wrapping positions is only part of periodic boundary behavior.
  Correct neighbor interactions across periodic seams need minimum-image
  distance and periodic cell lookup; this plan adds the wrap step and tests it,
  but only adds full periodic neighbor-distance handling if it remains small
  and localized.
- Leapfrog with summation-density WCSPH is a prototype integrator, not yet a
  full PySPH scheme/integrator replacement.

## Out of scope

- Public Cython ABI/API changes.
- Application/CLI integration.
- Full equation-codegen integration.
- Artificial viscosity, energy equation, XSPH, or adaptive timestep logic.
- Removing all remaining NNPS host readback for neighbor-array sizing.

## Estimated effort

One focused implementation session plus validation.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T12:08:27 CEST
- Approval, verbatim quote:
  > APPROVED
