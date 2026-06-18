---
type: plan
id: 2026-06-18_warp-grid-direct-neighbor-traversal
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-18T10:30:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, validation-benchmarks]
adr: ADR-0004
host_files:
  - pysph/base/warp_codegen.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Grid-direct neighbor traversal for the WCSPH continuity hot path

## Goal

Eliminate flat neighbor-list materialization (`build_neighbor_cache_gpu`: count
traversal -> exclusive scan -> ~4 MB host readback -> `wp.empty(total)` ->
fill traversal) from the continuity-density PEC path (ADR-0004). After ADR-0003
there is one fused equation launch per half-stage, so the flat list is built
only to be read once. Replace it by having the fused kernel walk the uniform
grid cell list directly. Per half-stage drops from {grid build + 2 build
traversals + readback + alloc + 1 consume traversal} to {grid build + 1
grid-direct traversal}.

Baseline (fused-equations slice, committed `429fa23e`): million-particle fixed
step `nx=565`, 10 steps -- cache build ~0.034-0.046 s/step is the dominant term;
best warm 10-step wall 3.77 s (`15.25x` vs CPU 57.48 s). Adaptive `nx=100`
resolved: exactly 1393 steps, Warp wall 10.83-14.33 s.

## Context

`build_neighbor_cache_gpu` (warp_nnps.py:963) builds a flat CSR list with two
neighbor traversals (`_grid_neighbor_lengths` at :305 to count,
`_grid_neighbor_fill` to fill) over the cell block produced by `_build_grid`
(:906, cached per `update()`). The continuity hot path has two neighbor
consumers: `compute_wcsph_accel_continuity` (the fused group kernel, always) and
`compute_wcsph_adaptive_timestep`'s `_wcsph_dt_factors` (warp_sph.py:1247, only
when `adaptive_dt`). The flat build also backs the summation path, the
per-equation oracle helpers, the host query API, and `compute_neighbor_sum` --
all off the continuity hot path and left untouched.

## Approach

### 1. Code generator (`pysph/base/warp_codegen.py`)

- Add `neighbor_mode='flat'|'grid'` to `generate_group_source` and
  `build_group_kernel` (default `'flat'` -- existing behavior, host query/oracle
  path, and all current tests unchanged). Thread the mode into `_cache_key`
  so flat and grid variants compile and cache independently.
- Grid signature replaces the flat `starts/lengths/neighbors` with the grid
  query arrays/bounds, matching `_grid_neighbor_lengths`:
  `cell_starts, cell_counts, cell_particles, xmin, ymin, zmin, cell_size,
  nx, ny, nz, ncells, radius_scale`. `dim`, `kernel_id`, equation scalars, and
  output arrays keep their positions after the neighbor section in both modes.
- Split `_emit_geometry` into a `phase` parameter: `pre` emits `dx,dy,dz,rij2`;
  `post` emits `rij,hij,grad,wij` and relative velocities; `all` (flat) emits
  the full sequence in the existing order (so flat output is byte-identical).
- Grid loop body: compute the destination cell index, iterate
  `dzc,dyc,dxc in range(-1,2)` with the in-bounds + `cid` checks, then
  `for pos in range(cell_starts[cid], cell_starts[cid]+cell_counts[cid])`,
  `j = wp.int32(cell_particles[pos])`, the `pre` geometry, the support cutoff
  `hi_ = radius_scale*d_h[i]; hj_ = radius_scale*s_h[j];
  if rij2 < hi_*hi_ or rij2 < hj_*hj_:` and -- inside the guard -- the `post`
  geometry then each block's `loop` snippet. `dx,dy,dz` are computed once and
  reused by the cutoff and the gradient. In grid mode force `x,y,z,h` into the
  signature even if the equations did not request them (the cutoff needs them).
- Re-indentation: the grid loop nests the body deeper than flat, so geometry
  lines and `loop` snippets (authored at the flat 8-space indent) are reindented
  by a fixed delta when emitted in grid mode. `initialize`/`post_loop` stay at
  indent 4 (outside the loop) in both modes. Add `neighbor_mode` to
  `GroupKernel` so the launcher binds the right inputs.

### 2. `warp_sph.py`

- `compute_wcsph_accel_continuity` grows `neighbor_mode='flat'`. In grid mode it
  calls `nnps._build_grid(src_index)` + reads `nnps._bounds`, builds the
  grid-mode group kernel, binds the grid arrays/bounds + `radius_scale` in
  signature order, does one launch + one sync, and never touches the flat cache.
- Add `_wcsph_dt_factors_{f32,f64}` grid-direct variants (hand kernels mirroring
  warp_sph.py:1247 with the cell-block loop + cutoff) and a grid path in
  `compute_wcsph_adaptive_timestep`.
- `_wc_sph_pec_continuity_step`: drop both `build_neighbor_cache_gpu` calls;
  build the grid once per half-stage (cached per `update()`), pass grid arrays
  to both consumers via the new grid paths.

### 3. Leave untouched

`build_neighbor_cache_gpu` and the flat path (summation density, per-equation
oracle helpers, host `get_nearest_particles`, `compute_neighbor_sum`), the
flat `_wcsph_dt_factors`, the Euler step, `_apply_wcsph_eos`.

## Files expected to change

- `pysph/base/warp_codegen.py`
- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_codegen.py`
- `pysph/base/tests/test_warp_sph.py`
- experiment.md + a new summary folder under
  `experiments/2026-06-16_warp-elliptical-drop-runner/`
- aspect contexts (warp-backend, gpu-nnps, validation-benchmarks), current.md,
  daily, session log, review artifact
- ADR-0004 registered in decisions/index.json + graph.md

## Tests / validation

- `test_warp_codegen.py`: a grid-mode generated kernel compiles and caches
  (mode is part of the key, distinct from flat); grid-mode vs flat-mode kernels
  produce identical outputs on a small multi-cell fixture sharing one cell list.
- `test_warp_sph.py`: the continuity half-stage calls `build_neighbor_cache_gpu`
  **zero** times on the continuity path (monkeypatch count) and issues the
  grid-direct launch; fused grid-direct vs separate-helper oracle parity on
  `arho, au, av, aw, ax, ay, az`; existing CPU PEC-parity test still passes.
- Adaptive `nx=100` resolved guard: still exactly 1393 steps; deltas at fp32
  scale vs committed Warp.
- Segmented million-particle (`nx=565`, 10 steps) profile: cache-build term -> 0
  on the continuity path; report new per-step wall and headline vs CPU 57.48 s
  and the prior best 3.77 s.
- `python .ai/.../scripts/validate-memory.py`; `git diff --check`.

## Success criteria

- The continuity path builds zero flat neighbor caches (test-asserted).
- Fused grid-direct matches the oracle and the CPU PEC state at fp32 scale; the
  adaptive `nx=100` run keeps exactly 1393 steps.
- The million-particle per-step wall drops materially with the cache-build term
  removed; reported in the review with the segmented profile.
- The focused suite passes (currently 47) plus the new grid tests.
- No multi-hour run is launched.

## Risks

- Membership parity: grid-direct must apply the same support cutoff the flat
  build applied, or it would include/exclude neighbors differently. Guarded by
  the grid-vs-flat and grid-vs-oracle parity tests.
- Register pressure / branch divergence: the cell walk inside the (now single)
  equation kernel may cut occupancy and offset the saved build. The segmented
  profile is the arbiter; fallback is ADR-0004's incremental-tuning alternative.
- fp32 neighbor-visitation-order change (cell order vs CSR order) shifts sums
  ~1e-7; bounded, and the adaptive step-count guard re-validates.
- Codegen indentation: the deeper grid nesting reindents snippets; a mistake
  surfaces as a Warp compile error. Mitigated by the grid-vs-flat parity test
  and keeping `initialize`/`post_loop` at the unchanged indent.

## Out of scope

- Migrating dt-factors / summation / oracle helpers onto the generator
  (ADR-0003 follow-up).
- Removing `build_neighbor_cache_gpu` entirely (still backs the host query API).
- Grid-direct for cross-array (`src != dst`) groups, periodic-distance support.
- Full `nx=565`, `tf=0.0076` GPU-only elliptical-drop run.

## Approval

- [x] Plan posted in chat and approved
- Approved by: @kunalpuri-prediqt at 2026-06-18T10:30:00 CEST
- Approval, verbatim quote:
  > APPROVED
