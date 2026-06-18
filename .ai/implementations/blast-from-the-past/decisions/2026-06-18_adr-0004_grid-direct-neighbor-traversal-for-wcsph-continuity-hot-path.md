---
type: decision
id: ADR-0004
date: 2026-06-18
author: @kunalpuri-prediqt
scope: gpu-nnps
status: Accepted
supersedes: []
relates_to: [ADR-0003]
depends_on: [ADR-0003]
conflicts_with: []
---

# ADR-0004: Grid-direct neighbor traversal for the WCSPH continuity hot path

## Context

After ADR-0003 fused the continuity-density PEC half-stage into a single
generated neighbor-loop kernel, the segmented million-particle profile
(`nx=565`, 1,002,885 particles, 10 fixed steps) showed the equation-kernel time
collapse (~0.064-0.088 s -> 0.011-0.014 s per step) and the **neighbor-cache
build** become the dominant per-step cost (~0.034-0.046 s, ~45-50% of the step
wall).

`UniformGridWarpNNPS.build_neighbor_cache_gpu` materializes a flat CSR neighbor
list per call: a **count** traversal of the 3x3(x3) cell block per destination
(`_grid_neighbor_lengths`), an exclusive `array_scan`, a host readback of the
full `lengths` array (~4 MB at 1M) plus a host `np.sum` to size the flat array
(forcing a device sync), a `wp.empty(total)` allocation, then a second **fill**
traversal of the same cells (`_grid_neighbor_fill`). The fused equation kernel
then walks that flat list a third time.

The flat list paid off when a half-stage launched 4-8 separate equation kernels
over it (build once, read many). After ADR-0003 there is a single fused
equation launch per half-stage, so the list is built (two traversals + readback
+ allocation) only to be read once.

Two consumers traverse neighbors on the continuity path: the fused equation
kernel (always) and the adaptive `_wcsph_dt_factors` CFL kernel (only when
`adaptive_dt` is on). The flat-list build also backs the summation-density
path, the per-equation oracle helpers, the host `get_nearest_particles` query
API, and `compute_neighbor_sum` -- none of which are on the continuity hot path.

## Decision

On the WCSPH continuity-density hot path, traverse the uniform-grid cell list
**directly** inside the consuming kernels instead of materializing a flat CSR
neighbor list.

The ADR-0003 code generator gains a `neighbor_mode='grid'` that emits the same
fused kernel body, but with the flat `starts/lengths/neighbors` loop replaced by
the cell-block iteration (`_build_grid` cell list + bounds), guarded by the same
support cutoff (`rij2 < (radius_scale*h_i)^2 or rij2 < (radius_scale*h_j)^2`) so
the kernel sees exactly the neighbor set the flat list contained. The shared
geometry is split: `dx,dy,dz,rij2` are computed before the cutoff; `rij,hij`,
the kernel gradient/value, and the relative velocities inside it. Equation
blocks are unchanged. The adaptive `_wcsph_dt_factors` kernel gets a hand-written
grid-direct variant (transitional, per ADR-0003's duplication stance).

`_wc_sph_pec_continuity_step` then builds only the grid (already cached per
`update()`), never the flat list. The grid build (cell-id/count kernel, scan,
scatter) stays -- it is the irreducible spatial index; what is removed is the
per-half-stage count traversal, scan, host readback, allocation, and fill
traversal.

## Rationale

- The flat list is now built to be read once; collapsing it into the consuming
  traversal removes two of the three per-half-stage neighbor traversals plus a
  forced host sync and a large allocation, directly attacking the dominant cost.
- `neighbor_mode` is a natural extension of the ADR-0003 generator: fusion of
  the *neighbor search* into the equation kernel, the same way equation fusion
  was a property of grouping. The flat path stays as the default for the host
  query API and the oracle, so nothing regresses.
- The support cutoff inside the grid loop reproduces the flat list's membership
  exactly, so generated-vs-oracle parity is preserved and testable.

## Alternatives considered

- **Incremental tuning of the flat build**: replace the 4 MB readback with a
  single scalar total from the scan tail, reuse `lengths/starts` device buffers
  across builds, single-pass fill. Lower risk but leaves all three traversals in
  place -- a modest win that does not remove the dominant term. Kept as the
  documented fallback if grid-direct regresses on the profile (register
  pressure / divergence from the cell walk inside the equation kernel).
- **Keep the flat list, persist it across the two half-stages within a step**:
  the coordinates move between half-stages (`update(push=False)` rebuilds), so a
  persisted list would be stale; rejected.
- **Defer**: the cache build is now the single largest per-step cost; deferring
  leaves the headline benchmark bottlenecked.

## Consequences

- A `neighbor_mode='grid'` code path in `warp_codegen.py` (signature, geometry
  split, cell-block loop, cutoff) and a grid-direct launch path in
  `compute_wcsph_accel_continuity`; the structural cache key gains the mode so
  flat and grid variants compile independently.
- A second representation of the CFL dt-factors kernel (flat + grid-direct)
  coexists transitionally, consistent with ADR-0003; the generator migration
  later absorbs it.
- fp32 accumulation order is unchanged versus the flat fused kernel (same block
  order, same per-pair math); the only numerical difference is neighbor
  *visitation order* within a destination (cell order vs CSR order), which can
  shift fp32 sums ~1e-7. Guarded by the adaptive `nx=100` resolved step-count
  check and generated-vs-oracle parity.
- Grid-direct iterates every occupant of the 27 cells and applies the cutoff
  per pair; the flat path applied the cutoff once at build time. The profile is
  the arbiter of whether the removed build/readback/alloc outweighs revisiting
  the cutoff inside the (now single) consuming traversal.

## Follow-ups

- ADR-0003 follow-up still stands: migrate the per-equation oracle helpers, the
  summation-density path, and the dt-factors kernel onto the generator and
  retire the duplicated hand kernels (the grid-direct dt-factors variant added
  here is part of what that migration absorbs).
- Once the continuity path no longer uses `build_neighbor_cache_gpu`, evaluate
  whether the flat-list build can be narrowed to the host query API / oracle and
  the summation path moved to grid-direct too.
