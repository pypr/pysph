---
type: review
date: 2026-06-18
user: @kunalpuri-prediqt
agent: claude
plan: .ai/implementations/blast-from-the-past/plans/2026-06-18_warp-grid-direct-neighbor-traversal.md
adrs: [ADR-0004]
aspects_touched: [warp-backend, gpu-nnps, validation-benchmarks]
host_files: [pysph/base/warp_codegen.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_codegen.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Warp grid-direct neighbor traversal for the WCSPH continuity hot path

## Diff summary

- `pysph/base/warp_codegen.py`: adds `neighbor_mode='flat'|'grid'` to
  `generate_group_source` / `build_group_kernel` (mode is part of `_cache_key`,
  exposed on `GroupKernel`). Grid mode emits the same fused body but replaces the
  flat `starts/lengths/neighbors` loop with a direct uniform-grid cell-list walk:
  - `_collect(neighbor_mode='grid')` forces `x,y,z` and `h` into the signature
    (the support cutoff needs them);
  - `_emit_geometry(..., phase=)` splits geometry into `pre` (`dx,dy,dz,rij2`)
    and `post` (`rij,hij,grad,wij,vel`); `phase='all'` reproduces the original
    flat ordering byte-for-byte;
  - the grid body computes the destination cell, iterates the 3x3(x3) block with
    the same in-bounds + `cid` checks as `_grid_neighbor_lengths`, applies the
    support cutoff `rij2 < (radius_scale*h_i)^2 or rij2 < (radius_scale*h_j)^2`,
    and runs the post geometry + each block's `loop` inside the cutoff guard;
  - `_reindent` shifts geometry/snippets to the deeper grid nesting;
    `initialize`/`post_loop` stay outside the loop, unchanged.
- `pysph/base/warp_sph.py`:
  - `_grid_launch_args(nnps, src_index, dtype)` binds the cell list (from
    `_build_grid`, cached per `update()`) + bounds + `radius_scale` in signature
    order;
  - `compute_wcsph_accel_continuity` grows `neighbor_mode='grid'` (default): a
    single grid-direct launch, no flat cache;
  - hand-written `_wcsph_dt_factors_grid_{f32,f64}` mirror the flat CFL
    dt-factors with the cell-block walk + cutoff; `compute_wcsph_adaptive_timestep`
    gains `neighbor_mode` (default `'flat'` -- summation path unchanged) and the
    continuity step passes `'grid'`;
  - `_wc_sph_pec_continuity_step` drops both `build_neighbor_cache_gpu` calls and
    builds only the grid.
- Tests: `test_warp_codegen.py` adds grid cache-distinct + forced-geometry and a
  single-cell numeric grid parity test; `test_warp_sph.py` adds grid-vs-flat
  fused parity and repurposes the old cache-reuse test to assert the continuity
  path builds zero flat caches (and that the grid is consulted).
- Docs: ADR-0004 (Accepted), plan, decision graph, experiment.md + summary
  folder `million-cpu-gpu-grid-direct/`, aspect contexts, current, daily,
  session log. New benchmark helper `profile_grid_direct_neighbors.py`.

## Aspects touched and host files modified

- Aspects: `gpu-nnps` (primary), `warp-backend`, `validation-benchmarks`.
- Host files: `pysph/base/warp_codegen.py`, `pysph/base/warp_sph.py`,
  `pysph/base/tests/test_warp_codegen.py`, `pysph/base/tests/test_warp_sph.py`.

## Behavioral / numerical changes

- The WCSPH continuity-density PEC path no longer materializes a flat CSR
  neighbor list. Both neighbor consumers (fused equations + adaptive CFL
  dt-factors) walk the cell list directly. Per half-stage: `{grid build + 2
  build-traversals + readback + alloc + 1 consume}` -> `{grid build + 1
  grid-direct traversal}`.
- The grid-direct loop applies the same support cutoff as
  `build_neighbor_cache_gpu` / `_grid_neighbor_lengths` (`radius_scale*h` on i
  and j; self-pair j==i included, contributing zero via the `rij>1e-12`/`grad`
  guards), so it visits exactly the flat list's neighbor set.
- Only the neighbor *visitation order* changes (cell order vs CSR order), which
  reorders fp32 sums ~1e-7. The per-pair math and block order are identical to
  the flat fused kernel; the CFL factor is an order-independent `max`, dt_force
  is neighbor-independent.
- Retained on the flat path (unchanged): the summation-density path, the
  per-equation oracle helpers, host `get_nearest_particles`,
  `compute_neighbor_sum`, the flat `_wcsph_dt_factors`, and
  `compute_wcsph_adaptive_timestep`'s default (`'flat'`).
- Million-particle fixed-step (`nx=565`, 1,002,885 particles, segmented, 2
  warmup discarded): `build_neighbor_cache_gpu` called 0 times on the continuity
  path; cache-build term ~0.034-0.046 s/step -> 0; grid build ~0.0004-0.0007 s;
  equation kernel 0.011-0.014 -> 0.023-0.025 s/launch (absorbs the cutoff
  traversal); step wall (steady) 0.076-0.098 -> 0.059-0.064 s (~25-35% lower);
  KE delta vs flat fused `-1.99e-06`, all finite.
- Adaptive `nx=100` resolved guard (Warp-only, pysph timestep policy): `1393`
  steps (identical to committed), all finite; deltas vs committed Warp
  axis_major `2.4e-07`, axis_minor `6.6e-07`, rho `~9e-07`, kinetic_energy
  `1.19e-04` (relative `1.5e-08`); Warp wall `7.82 s` (committed `23.63 s`).

## Tests / validation run

```text
$ python -m py_compile pysph/base/warp_codegen.py pysph/base/warp_sph.py pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py
pass
```

```text
$ python -m pytest -q pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
50 passed, 2 warnings
```

```text
$ PYTHONPATH=.ai/.../2026-06-16_warp-elliptical-drop-runner \
  python .ai/.../profile_grid_direct_neighbors.py --nx 565 --steps 12 --warmup 2
flat_cache_builds_total 0; grid_builds/step 2.0; equation launches/step 2
step_wall_s steady [0.059156, 0.063873]; grid_build_s steady [0.000379, 0.000692]
equation_launch_s steady/launch [0.022992, 0.025071]; kinetic_energy 7854.1276; all_finite True
```

```text
$ adaptive nx=100 resolved guard (Warp-only, pysph timestep policy, grid-direct)
steps 1393 (committed 1393), warp wall 7.82 s, all_finite True
deltas vs committed Warp: axis_major 2.384e-07, axis_minor 6.557e-07,
rho_min -8.941e-07, rho_max 2.384e-07, kinetic_energy 1.19e-04 (rel 1.5e-08)
```

Fresh same-session CPU-vs-Warp headlines (no reused numbers; CPU =
single-threaded PySPH Cython Application, Warp = grid-direct on RTX 4060 fp32):

```text
$ nx=100 resolved (real PySPH Application vs Warp, adaptive, identical 1393 steps)
CPU 160.10 s (0.1149 s/step) | Warp 6.78 s (0.00487 s/step) | speedup 23.6x

$ 1M particles, 100 fixed steps (n_damp=0, identical dt both sides)
CPU 344.50 s (3.445 s/step) | Warp 8.37 s total (2.39 setup + 5.98 step; 0.0598 s/step)
speedup 41.2x wall / 57.6x per-step | KE rel delta 1.6e-09 | all_finite True
```

## validate-memory.py

```text
validate-memory: PASS
```

```text
$ git diff --check -- pysph/base/warp_codegen.py pysph/base/warp_sph.py pysph/base/tests .ai/implementations/blast-from-the-past
clean (to be re-run before commit)
```

## Adversarial pre-review

Before this review, a 6-dimension adversarial workflow (membership parity,
arg-binding order, codegen indentation, flat-path regression, edge cases,
numerical/fp32) reviewed the diff; each finding was independently verified by a
skeptic. Result: 1 confirmed finding (a nit), 0 others.

- Confirmed nit: defaulting `compute_wcsph_adaptive_timestep` to
  `neighbor_mode='grid'` had silently re-routed the *summation-density* leapfrog
  step's adaptive-dt to grid (verified numerically benign -- identical
  membership, order-independent max), but that path is out of ADR-0004's stated
  scope. **Addressed**: the helper now defaults to `'flat'` (summation path
  byte-identical to commit `429fa23e`) and the continuity step passes `'grid'`
  explicitly. Suite re-run green (50 passed).

## Boundary amendment

- implementation.md boundary section: n-a (`pysph/base/warp_*.py` and
  `pysph/base/tests/test_warp_*.py` are already in the boundary).
- Amendments log entry: n-a

## Visual aid

Segmented per-step comparison (million-particle fixed step, steady state):

```text
                       flat fused (ADR-0003)        grid-direct (ADR-0004)
flat cache builds/step   ##  (2)                      .   (0)
flat cache build/step    ~0.034-0.046 s               0 (removed)
grid build/step          (inside cache build)         ~0.0008-0.0014 s
equation time/step       0.011-0.014 s                0.046-0.050 s  (absorbs traversal)
neighbor traversals/half ### (count+fill+read = 3)    #  (1: cutoff walk)
step wall/step           0.076-0.098 s                0.059-0.064 s  (~25-35% lower)
```

## Risks

- fp32 visitation-order change shifts sums ~1e-7 (KE ~1e-4 absolute / ~1.5e-8
  relative in the adaptive run); within parity tolerances. The adaptive guard
  confirms identical step count (1393).
- Grid-direct iterates every occupant of the 27 cells and applies the cutoff per
  pair; the flat path applied it once at build. The profile is the arbiter: the
  removed build/readback/alloc outweighs the cutoff revisit, netting ~25-35%
  lower per-step wall, though the equation kernel itself is ~2x slower per
  launch.
- Grid mode requires a grid-capable NNPS (`_build_grid`/`_bounds`).
  `UniformGridWarpNNPS` is the only NNPS on the WCSPH path; a `BruteForceWarpNNPS`
  would raise on grid mode, but that path is not reachable from the WCSPH steps.
- Transitional duplication: `_wcsph_dt_factors` now exists in flat and
  grid-direct forms. Consistent with ADR-0003; the generator migration absorbs
  both.

## Unresolved questions

- Should the summation-density path also move to grid-direct (it currently still
  builds a flat cache via summation density)? Deferred; the ADR-0004 follow-up
  evaluates narrowing `build_neighbor_cache_gpu` to the host-query/oracle path.
- The equation kernel is now ~2x slower per launch (it does the cutoff walk).
  Worth exploring a one-pass count-free CSR or shared-memory cell cooperation if
  the neighbor traversal becomes the next bottleneck.

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > LGTM
