---
type: plan
id: 2026-06-16_warp-elliptical-drop-application-runner
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T12:45:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: []
within_boundary: true
---

# Plan: Warp elliptical drop application runner

## Goal

Add a runnable Warp elliptical-drop-style application experiment that creates
the standard elliptical-drop initial particle patch, advances it with the
current Warp NNPS + WCSPH leapfrog prototype, and records enough metrics to
decide the next physics/integration work.

## Context

The existing PySPH `pysph.examples.elliptical_drop` application uses the normal
PySPH solver path with `WCSPHScheme`, Gaussian kernel, EPEC integrator, Tait
EOS, artificial viscosity, and XSPH correction. The current Warp prototype does
not yet implement that full formulation. It has:

- `UniformGridWarpNNPS`;
- CubicSpline summation density;
- isothermal EOS;
- inviscid pressure-gradient acceleration;
- KDK leapfrog;
- device-side periodic position wrapping, but not periodic neighbor distances.

So this plan creates a first GPU runner/smoke application, not a validated
drop-physics result. The runner should make the gap explicit and produce data
that guides the next equation work.

## Approach

1. Create a new experiment packet under
   `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`.
2. Add a Python runner script with an application-style class, for example
   `WarpEllipticalDropRunner`, that:
   - accepts `--nx`, `--steps`, `--dt`, `--rho0`, `--c0`, `--p0`, and output
     path options;
   - creates the elliptical-drop initial circular patch using the same geometry
     and initial velocity field as the PySPH example;
   - initializes required Warp properties (`rho`, `p`, `au`, `av`, `aw`);
   - advances with `UniformGridWarpNNPS` and `wc_sph_leapfrog_step`;
   - pulls final arrays once at the end;
   - writes a compact `.npz` result with final state and scalar metrics such as
     particle count, final time, min/max coordinate bounds, kinetic energy, and
     finite-value checks.
3. Add `run_correctness.sh` for a small smoke run, likely `nx=8` or `nx=10`
   with a tiny number of steps, to keep it stable and quick.
4. Add experiment documentation with:
   - what to expect;
   - success criteria;
   - explicit note that this is not the full PySPH elliptical-drop physics yet;
   - next missing equations for a faithful run.
5. Run the wrapper and record raw output.

## Files expected to change

- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py`
- `.ai/implementations/blast-from-the-past/current.md`
- relevant aspect/session/daily memory updates

No host package file is expected to change in this first runner slice.

## Tests / validation

- `bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai/implementations/blast-from-the-past`

## Risks

- The current Warp formulation is incomplete for the published elliptical-drop
  benchmark, so success criteria must focus on runnable GPU state evolution,
  finite values, and output generation.
- The pressure-gradient-only dynamics may be physically rough or unstable for
  aggressive `nx`, `dt`, or step counts.
- This does not yet exercise PySPH's `Application`/`Solver` plumbing directly;
  it is an application-style experiment runner around the Warp prototype.

## Out of scope

- Full PySPH `Application` CLI integration.
- Matching the analytical elliptical-drop locus.
- Tait EOS, artificial viscosity, XSPH, Gaussian kernel support, or adaptive
  timestep.
- PR creation.

## Estimated effort

One focused implementation session.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T12:30:36 CEST
- Approval, verbatim quote:
  > APPROVED
