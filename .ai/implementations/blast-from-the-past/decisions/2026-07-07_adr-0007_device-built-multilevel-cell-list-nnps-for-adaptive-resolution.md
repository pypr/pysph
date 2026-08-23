---
type: decision
id: ADR-0007
date: 2026-07-07
author: @kunalpuri-prediqt
scope: gpu-nnps
status: Proposed
supersedes: []
relates_to: [ADR-0003, ADR-0004]
depends_on: [ADR-0004]
conflicts_with: []
---

# ADR-0007: device-built multilevel cell-list NNPS for adaptive resolution

## Context

`UniformGridWarpNNPS` sizes one cell (`cell_size = radius_scale * global_hmax`)
for the whole domain. That is correct for variable `h` but not scalable for
adaptive particle resolution (APR): a small number of coarse particles makes
cells coarse everywhere, so a dense fine region generates many candidate checks
per destination. The plan
`2026-07-06_warp-multilevel-gpu-nnps` set the checkpoint gate: an exact,
device-built multilevel cell-list NNPS whose accepted neighbor set matches
brute force exactly while candidate work drops materially on localized
refinement, and whose construction stays on the GPU. This ADR records the
representation and contracts chosen once the exact-set and candidate-scaling
kill tests passed; it is narrowed (per the plan) to the NNPS only. Runtime
split/merge, particle pools, target-level assignment, and periodic multilevel
domains are explicitly out of scope and receive later ADRs.

The P0 Vacondio/PySPH daughter-stencil convention mismatch is unresolved, so
the level ratios here are provisional configuration, not a production default.

## Decision

Add `MultilevelGridWarpNNPS(UniformGridWarpNNPS)` as an additive prototype
alongside the uniform grid, with these choices:

- **Level representation.** Discrete smoothing-length levels via `h_ref`,
  `level_ratio`, `nlevels`. Level `k` covers the half-open range
  `[h_ref*level_ratio**k, h_ref*level_ratio**(k+1))` with level 0 the finest;
  the overall top edge is inclusive. Each source particle belongs to exactly
  one level (so no pair is visited twice). Level edges are computed in the
  device float precision so an `h` sitting on an edge in fp32 bins like the
  edge rather than tripping the range guard by one ULP. `h` outside
  `[edges[0], edges[nlevels]]` fails loudly; particles are never silently
  clipped. Each populated level's conservative support bound is
  `radius_scale * max(h in level)`.

- **Flattened per-level grids.** Each populated level gets its own padded
  origin (min-coordinate minus one cell), cell size (its support bound), and
  `(nx,ny,nz)`. All levels are flattened into one global cell space via a
  per-level `cell_offset`. The grid is built with the existing
  count -> exclusive-scan -> scatter kernels over that global cell space. Empty
  levels allocate no cells (`nx=0`) and are skipped.

- **Device-built construction / permitted readback.** Level assignment and the
  per-level count, max-`h`, and AABB (min/max per axis) reductions run in one
  GPU kernel (`_ml_assign_reduce`). Only `O(nlevels * narrays)` scalar metadata
  is read back to the host to size the dense level grids; per-particle
  `x/y/z/h` never leave the device on a warm `update(push=False)` or during
  traversal. Per-particle levels stay on the device; the host-facing
  `level_grid_info` diagnostic reads them back lazily and is not on the query
  path.

- **Cross-level traversal.** For each destination and each level, the query
  radius is `max(radius_scale*h_i, level_support[k])`. That radius is converted
  to the necessary cell-index range for that level (a variable range, not a
  fixed 3x3x3 stencil, with a +/-1 guard band), then the exact pairwise
  symmetric cutoff `rij^2 < (radius_scale*h_i)^2 OR rij^2 < (radius_scale*h_j)^2`
  is applied before a pair is accepted. The length and fill passes are
  structurally identical so their counts can never diverge.

- **Module isolation.** The multilevel `@wp.kernel`s and the class live in their
  own module `pysph/base/warp_multilevel_nnps.py`, not in `warp_nnps.py`. Warp
  compiles/loads an entire Python module's kernels together on first launch, so
  co-locating them with the uniform-grid kernels would JIT them onto the device
  for every WCSPH consumer of `UniformGridWarpNNPS` and inflate the process
  module footprint. Isolation loads them only when the multilevel NNPS is used.

Provisional configuration: initial tests use ratios 2 and 1.2 and up to four
levels spanning `h_max/h_min = 16`. No production ratio or split weight is
selected here.

## Rationale

- Per-level cell sizing removes the artificial candidate work that a single
  global-`hmax` grid imposes on dense fine regions, while the exact symmetric
  cutoff inside the traversal reproduces the brute-force accepted set, so
  correctness is preserved and testable against the (independently verified)
  numpy oracle and `BruteForceWarpNNPS`.
- Exactly-one-level membership plus non-overlapping per-level cell ranges give
  the no-duplicate-visit guarantee for free.
- On-device assignment + reductions with only `O(nlevels)` readback keeps the
  structure device-resident as APR requires, and is measurable/enforced by a
  residency test that fails if `x/y/z/h` are pulled to the host.
- Housing the kernels in a separate module keeps the WCSPH hot path's device
  footprint unchanged, which matters because the in-process PTX-JIT on the
  WSL2 dev box is already near its module-accumulation ceiling.

## Alternatives considered

- **Keep the global-`hmax` uniform grid.** Correct but not scalable for APR:
  the candidate-scaling kill test exists precisely because coarse particles
  over-coarsen the whole grid.
- **Sparse hash / sort-based level storage** instead of dense per-level grids.
  Retained as the documented fallback (see decision gate below): if dense
  per-level grids consume more memory than the saved particle state, or exact
  traversal needs unbounded work, on representative localized-refinement
  distributions, switch to sparse storage before this ADR is Accepted.
- **Multilevel kernels inside `warp_nnps`.** Rejected: it hangs the combined
  single-process WCSPH suite on the WSL2 PTX-JIT (module-footprint
  accumulation); the separate-module form does not.
- **Host-side level assignment / AABB** (read `x/y/z/h` back, bin on the host).
  Simpler, and was used as a transitional first cut, but violates the
  device-residency gate; replaced by the on-device assign+reduce kernel.

## Consequences

- New module `pysph/base/warp_multilevel_nnps.py`: the assign+reduce kernel,
  the flattened cell-id / variable-stencil length+fill traversal kernels
  (fp32+fp64), `MultilevelGridWarpNNPS`, a device neighbor-cache oracle
  (`build_neighbor_cache_gpu`), and host diagnostics (`level_grid_info`,
  `candidate_pairs`). Pure-Python helpers (`assign_particle_levels`,
  `brute_force_neighbor_sets`, `accepted_level_pair_counts`) live in
  `warp_nnps.py` (no GPU code).
- Kill-gate fixtures (all passing) in `test_warp_nnps.py`: single-level parity,
  four-level `h_max/h_min=16` cross-level parity + level-pair matrix, fp32
  per-level grid-boundary padding, empty interior levels, gradual ratio 1.2,
  cross-array traversal + per-source ownership, particles at spatial bounds,
  device residency (no coordinate host readback), and clustered refinement
  (accepted sets identical to the oracle and the uniform grid; candidate work
  `>= 4x` lower -- measured ~9x, 197k vs 1.77M candidate pairs on the fixture).
- Cross-scale work that is real is not removed: a coarse destination querying a
  dense fine level legitimately spans many fine cells. The hierarchy removes
  artificial fine-fine and empty-cell work, not real cross-scale neighbors.
- The permitted `O(nlevels*narrays)` metadata readback is a per-`update()` host
  synchronization; it is a prototype allowance, NOT the production
  device-residency contract. The eventual production APR path is expected to
  eliminate it (e.g. a persistent max-levels allocation) and this constraint
  must not be silently inherited.
- Periodic multilevel traversal is out of scope; it must raise a clear error
  rather than silently use the non-periodic walk. (The generated multilevel
  equation loop -- plan step 3 -- will carry that guard.)

## Follow-ups

- Plan step 3: `neighbor_mode='multilevel'` in `warp_codegen.py` and routing of
  `_run_equation_group` / `compute_wcsph_adaptive_timestep` so real SPH equation
  groups and the adaptive-timestep reduction consume the multilevel structure
  directly, with fp32/fp64 output parity against the flat/uniform path.
- Reject periodic multilevel domains explicitly with an error in step 3.
- Decision gate before this ADR moves to Accepted: confirm on representative
  localized-refinement cases that dense per-level grids do not consume more
  memory than the saved particle state; otherwise adopt sparse hash/sort
  storage.
- Eliminate the per-`update()` metadata readback for the production path.
- Select production level ratios / split policy only after the P0 stencil
  convention mismatch is resolved (later ADRs).
