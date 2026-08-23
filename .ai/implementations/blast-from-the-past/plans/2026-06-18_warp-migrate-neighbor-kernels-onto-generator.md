---
type: plan
id: 2026-06-18_warp-migrate-neighbor-kernels-onto-generator
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-18T12:30:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps]
adr: ADR-0003
within_boundary: true
host_files:
  - pysph/base/warp_codegen.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
---

# Plan: Migrate remaining neighbor-loop kernels onto the generator (ADR-0003 follow-up)

## Goal

Make `warp_codegen` the single source for every neighbor-loop kernel and retire
the duplicated hand-written `@wp.kernel`s. Behavior-preserving consolidation
(perf-neutral): the standalone equation helpers keep their public signatures,
I/O, cross-array support, and flat default; only the kernel they launch changes
from a hand kernel to a generated single-block group. This is ADR-0003's
recorded follow-up; no new ADR.

Not in scope: fusing/grid-directing the summation path, narrowing
`build_neighbor_cache_gpu`, periodic distance.

## Context

Hand kernels duplicating generator blocks (to retire, ~14): `_summation_density`,
`_continuity`, `_pressure_gradient`, `_artificial_viscosity`,
`_xsph_correction` (f32+f64 each), and `_wcsph_dt_factors{,_grid}` (f32+f64).

Key semantic detail: the standalone helpers compose via **read-modify-write**.
`_artificial_viscosity` does `au = d_au[i]; au += ...; d_au[i] = au` so the
summation path (`_compute_wcsph_acceleration`: pressure gradient then viscosity,
two launches) accumulates viscosity onto the pressure-gradient acceleration.
Pressure gradient / continuity / xsph / summation density **overwrite** their
outputs (first/only contribution). Generator blocks currently overwrite (write a
shared `_acc` once), which is correct for the fused continuity group but would
clobber for the standalone additive viscosity.

The CFL dt-factor is a per-particle **max-reduction** over neighbors plus a
neighbor-independent `dt_force`; both fit the generator because block `loop`
snippets are free-form (`_acc_dt_cfl = wp.max(...)`).

Tests call the public helpers, never the inner kernels; the standalone helpers
have cross-array tests (`src != dst`). The generator already emits `s_`/`d_`
arrays separately, so cross-array is preserved.

## Approach

### 1. Generator: add `accumulate_outputs` (`warp_codegen.py`)

- `generate_group_source(..., accumulate_outputs=False)`: when True, initialize
  each accumulator from the existing output (`_acc_<out> = d_<out>[i]`) instead
  of `TYPE(0.0)`, so the group adds to the destination arrays (read-modify-write).
- Thread through `build_group_kernel`; include in `_cache_key` and `GroupKernel`.

### 2. New blocks (`warp_sph.py`)

- `SummationDensity` (`src_arrays=('m',)`, `out_arrays=('rho',)`,
  `requires=('rij','hij','wij')`, `loop: _acc_rho += s_m[j]*wij`).
- `WcsphCflFactor` (`dst_arrays=('au','av','aw')`, `out_arrays=('dt_cfl','dt_force')`,
  `scalars=('c0',)`, `requires=('dx','dy','dz','rij2','hij','vij*')`):
  `loop` does the `rij2>1e-12` guard + `_acc_dt_cfl = wp.max(_acc_dt_cfl,
  |hij*vdotx/rij2| + c0)`; `post_loop` sets `d_dt_force[i] = au^2+av^2+aw^2`.
  Runs in flat and grid via `neighbor_mode` (de-duplicates both dt-factors
  kernels into one block).

### 3. Shared launcher + repoint helpers

- Add `_run_equation_group(nnps, src_index, dst_index, blocks, scalar_values,
  kernel, cache, neighbor_mode, accumulate_outputs)` that builds the group, binds
  inputs in the generator's canonical order (src arrays, dst arrays, neighbor
  section [flat cache or `_grid_launch_args`], dim, kernel_id, scalars, outputs),
  launches once, syncs.
- Repoint, preserving each public helper's signature/push/return/cross-array and
  **flat default**:
  - `compute_summation_density` -> `[SummationDensity()]`, overwrite.
  - `compute_continuity` -> `[ContinuityEquation()]`, overwrite.
  - `compute_pressure_gradient` -> `[PressureGradient()]`, overwrite.
  - `compute_artificial_viscosity` -> `[ArtificialViscosity()]`, **accumulate=True**.
  - `compute_xsph_correction` -> `[XSPHCorrection()]`, overwrite.
  - `compute_wcsph_adaptive_timestep` -> `[WcsphCflFactor()]` (flat default;
    continuity step keeps passing `grid`), then the existing init/reduce/finalize
    reductions unchanged.
- Refactor `compute_wcsph_accel_continuity` to use `_run_equation_group` too
  (DRY; same fused group, grid default).
- `_compute_wcsph_acceleration` (summation path) is unchanged in structure:
  pressure gradient (overwrite) then viscosity (accumulate) then continuity, now
  generator-backed -- byte-for-byte same composition.

### 4. Retire the hand kernels

Delete the ~14 duplicated `@wp.kernel`s. Keep the EOS kernels (per-particle, no
neighbor loop), the integrator kernels, and the dt init/reduce/finalize
reduction kernels.

## Tests / validation

- Full focused suite stays green (every helper is now generator-backed and is
  covered by the existing CPU-reference parity + cross-array tests; the additive
  viscosity test guards `accumulate=True`).
- `test_warp_fused_accel_matches_separate_helpers` becomes a fusion-consistency
  check (both sides generator-backed); kept.
- New `test_warp_codegen.py` test: `accumulate_outputs=True` adds to the existing
  output (vs overwrite for False) on a small fixture.
- Adaptive `nx=100` resolved guard keeps exactly `1393` steps (validates the
  migrated dt-factors in grid mode).
- Quick million fixed-step profile: continuity path unchanged, no regression.
- `validate-memory.py`; `git diff --check`; then an adversarial-review workflow
  over the diff before sign-off.

## Success criteria

- The ~14 hand neighbor-loop kernels are gone; the generator is the single
  source. Focused suite passes; adaptive guard keeps 1393 steps; CPU-parity and
  cross-array deltas stay at fp32 scale; no million-particle regression.

## Risks

- Additive vs overwrite semantics: viscosity must stay additive
  (`accumulate=True`); guarded by `test_warp_artificial_viscosity_matches_cpu_and_adds_to_acceleration`.
- I/O-contract drift in the rewritten helpers (push/pull/return/cross-array);
  guarded by the per-helper CPU + cross-array tests.
- fp32 single-block-group vs hand-kernel ordering ~1e-7; within parity tolerances.
- CFL `max` via free-form snippet relies on `_acc` init 0 being a valid identity
  (factors > 0); true here.

## Approval

- [x] Plan posted in chat and approved
- Approved by: @kunalpuri-prediqt at 2026-06-18T12:30:00 CEST
- Approval, verbatim quote:
  > APPROVED
