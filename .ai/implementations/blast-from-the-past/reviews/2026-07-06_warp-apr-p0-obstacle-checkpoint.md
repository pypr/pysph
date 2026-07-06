---
type: review
date: 2026-07-06
user: @kunalpuri-prediqt
agent: codex
plan: 2026-07-06_warp-dynamic-adaptive-particle-resolution
adrs: [ADR-0005]
aspects_touched: [warp-backend, validation-benchmarks, host-integration]
host_files: []
review_mode: prototype-owner
status: prototype-approved
---

# Review - APR P0 fixed-obstacle checkpoint

## Diff summary

- Extend the existing experiment runner with opt-in `--with-obstacle` support.
  It consumes the third array already produced by `DamBreak3DGeometry` and
  passes all solids to the existing `wc_sph_dam_break_step`; no backend changes.
- Generalize runner metrics/output for an optional obstacle while preserving
  no-obstacle behavior.
- Add three assessed primary APR reference notes and index entries.
- Open the P0 experiment and record one-step, startup, and first-impact NPZ
  outputs plus exact metrics.
- Add a NumPy-only 3D split-stencil density minimization and its 101-grid JSON
  output; record the converged paper/PySPH discrepancy as a kill-test failure.
- Update current/session/daily memory with the active P0 checkpoint.

## Aspects touched and host files modified

Experiment/memory only. No host PySPH file is modified.

## Behavioral / numerical changes

The runner has a new opt-in fixed obstacle. Default `with_obstacle=False`
preserves the original two-array case. With the option enabled, the runner uses
fluid index 0 and every remaining array as a fixed solid source.

The 250-step coarse run develops fluid/wall/obstacle pressure and reaches the
obstacle while keeping its device coordinates unchanged. This is prototype
evidence at `dx=0.10`, not a converged Kleefsman pressure claim.

## Tests / validation run

```text
$ python -m py_compile .../dam_break_3d_runner.py
<no output; exit 0>

$ python .../dam_break_3d_runner.py --dx 0.1 --steps 1
fluid/wall = 1000/3824; obstacle=0; all_finite=true; exit 0

$ python .../dam_break_3d_runner.py --dx 0.1 --steps 20 --with-obstacle
fluid/wall/obstacle = 1000/3824/4
t=0.003095018; all_finite=true; elapsed=3.16 s

$ python .../dam_break_3d_runner.py --dx 0.1 --steps 250 --with-obstacle
fluid/wall/obstacle = 1000/3824/4
t=0.258454926; all_finite=true
rho=983.44897..1022.19061 kg/m^3
obstacle_p=23.696..150.147 kPa
surge_front_x=2.4901464; elapsed=5.76 s; max RSS=350756 KiB

$ compare obstacle x/y/z at step 1 and step 250
obstacle_device_position_drift=0

$ python split_stencil_density_kill.py --ngrid 101
cubic+center: E=1.2586503e-3, mass sum=1
icosa+center: E=3.5803204e-4, mass sum=1, min/max=0.6565656
paper Table 1: E=8.326e-5, min/max=0.33

$ python split_stencil_density_kill.py --ngrid 81/121
icosa E=3.5803275e-4 / 3.5803200e-4 (quadrature converged)

$ git diff --check
<no output; exit 0>
```

## validate-memory.py

```text
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: n-a
- Amendments log entry: n-a

Only `.ai/` experiment and implementation-memory files changed.

## Visual aid

Waiver: this checkpoint proves array wiring and numerical finiteness; no visual
scientific claim is made. The committed NPZ contains fluid, wall, and obstacle
coordinates/fields for later checkpoint rendering.

## Risks

- Four obstacle particles are sufficient for wiring/impact smoke but far too
  coarse for validation against experimental pressure traces.
- The first cold execution paid multi-minute generated Wendland cache loads;
  warm timings exclude that one-time compilation cost.
- Pointwise pressure is not converged and may be fp32/noise sensitive. P0 still
  needs probe impulse and uniform coarse/fine comparisons.
- The Muta--Ramachandran implementation is 2D-first; treating it as a 3D GPU
  implementation would be incorrect.
- The first faithful-looking Vacondio reproduction does not match its numeric
  table under PySPH's kernel convention. No production split weights may be
  chosen from this result until the discrepancy is explained.

## Unresolved questions

- Vacondio's `epsilon=0.65`, `alpha=0.70`, 12-shell-plus-center stencil is now
  ingested, but its Table 1 mass ratio/error is not reproduced by the
  constrained PySPH-Wendland calculation.
- The tradeoff between a 14+center split and cheaper binary/eight-child GPU
  refinement remains undecided; ADR-0007 is intentionally deferred.

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > commit this prototype checkpoint. list out next steps
- Timestamp: 2026-07-06T11:00:43 CEST
