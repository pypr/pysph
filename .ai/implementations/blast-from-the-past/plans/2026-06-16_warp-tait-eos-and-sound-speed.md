---
type: plan
id: 2026-06-16_warp-tait-eos-and-sound-speed
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T14:05:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
within_boundary: false
---

# Plan: Warp Tait EOS and Per-Particle Sound Speed

## Goal

Add the PySPH WCSPH Tait equation of state to the Warp prototype, compute
per-particle sound speed `cs`, and use pair-averaged `cs` in the artificial
viscosity term. This removes the constant-`c0` approximation from the current
Warp momentum path.

## Context

PySPH's `pysph.sph.wc.basic.TaitEOS` computes:

```text
ratio = rho / rho0
p = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1)
cs = c0 * ratio**(0.5*(gamma - 1))
```

PySPH's WCSPH `MomentumEquation` uses:

```text
cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])
Pi_ij = (-alpha*cij*mu_ij + beta*mu_ij^2) * RHOIJ1
```

The committed Warp artificial-viscosity slice currently uses constant `c0` for
`cij`. Tait EOS plus `cs` should make that term match PySPH more closely.

## Production Elliptical-Drop Readiness

This slice will not make the Warp elliptical-drop runner production-ready by
itself. It should upgrade the physics smoke run, but production/published
elliptical drop still needs:

- Gaussian kernel support or an accepted kernel-equivalence decision.
- XSPH correction.
- Adaptive timestep and CFL/dt tracking.
- Baseline comparison against PySPH's elliptical-drop outputs at the named
  times.
- Clear correctness/timing thresholds for accepting the run as a benchmark.
- Application/Solver integration or an explicit decision that the standalone
  runner is sufficient for the first benchmark.

## Approach

1. Add Warp float32/float64 Tait EOS kernels that write both `p` and `cs`.
2. Add `compute_tait_eos(pa, rho0, c0, gamma=7.0, p0=0.0, ...)`.
3. Keep `compute_isothermal_eos()` intact so earlier tests and comparisons
   remain valid.
4. Update artificial-viscosity kernels to accept source/destination `cs` arrays
   and use `cij = 0.5*(d_cs + s_cs)` instead of constant `c0`.
5. Update `_compute_wcsph_acceleration()`, `wc_sph_leapfrog_step()`, and
   `wc_sph_euler_step()` with an EOS mode or boolean that defaults to current
   isothermal behavior, while allowing `eos='tait'`.
6. Update the elliptical-drop runner to default to Tait EOS with `gamma=7.0`
   and ensure the particle array carries `cs`.
7. Add tests:
   - direct Tait EOS pressure and `cs` against CPU reference;
   - artificial viscosity using pair-averaged `cs`;
   - a small WCSPH step with Tait EOS where final `p`, `cs`, and acceleration
     match CPU reference.
8. Run the focused Warp SPH/NNPS suite and the elliptical-drop smoke wrapper.
9. Update experiment and memory files with the new results.

## Files expected to change

- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_sph.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
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
- Optional small ramp: `nx=16`, `steps=5`, `dt=1.0e-5`, Tait EOS, `alpha=0.1`.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past`

## Risks

- Negative or very low density ratios can make fractional-power `cs` invalid;
  the focused tests and smoke should watch for non-finite values.
- Switching the runner default from isothermal to Tait changes smoke metrics.
- Momentum correctness will still be incomplete until XSPH and Gaussian kernel
  support land.

## Out of scope

- XSPH correction.
- Gaussian kernel support.
- Adaptive timestep integration.
- Production benchmark acceptance thresholds.
- PR creation.

## Estimated effort

One focused implementation session after approval.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T13:57:05 CEST
- Approval, verbatim quote:
  > APPROVED
