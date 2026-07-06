---
type: plan
id: 2026-07-06_warp-multilevel-gpu-nnps
author: @kunalpuri-prediqt
agent: codex
created: 2026-07-06T11:18:54 CEST
status: draft
aspects: [gpu-nnps, warp-backend, particle-memory, validation-benchmarks]
host_files:
  - pysph/base/warp_nnps.py
  - pysph/base/warp_codegen.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_nnps.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp multilevel GPU NNPS for adaptive resolution

## Goal

Build the first runtime GPU foundation for adaptive particle resolution: an
exact, device-built multilevel cell-list NNPS that efficiently handles discrete
particle smoothing-length levels without sizing every cell from global
`h_max`.

The checkpoint ends when existing generated SPH equation groups and the
adaptive-timestep reduction can consume the multilevel structure directly,
mixed-resolution neighbor sets match brute force exactly, and a representative
localized-refinement fixture shows materially less candidate work than the
current global-`h_max` uniform grid.

This plan does **not** port the offline split-stencil minimization to the GPU.
That calculation produces constants before a simulation starts. It also does
not implement runtime split/merge or particle allocation; those consume this
NNPS in the next approved checkpoint.

## Context

`UniformGridWarpNNPS` currently:

- pulls every array's `x/y/z/h` to the host in
  `_compute_bounds_and_cell_size()`;
- chooses one `cell_size = radius_scale * global_hmax`;
- builds a device count/scan/scatter cell list per source array;
- feeds the generated `neighbor_mode='grid'` loop directly, with the exact
  symmetric inclusion rule `rij < radius_scale*h_i OR ...*h_j`.

That is correct for variable `h`, but not scalable for APR. A small number of
coarse particles makes cells coarse everywhere, so dense fine regions generate
many candidate checks. The current generated kernel also assumes one scalar
origin, cell size, and `(nx, ny, nz)` tuple.

ADR-0004 adopted grid-direct traversal because each fused equation group reads
the neighbor structure once. This plan preserves that decision and adds a new
structural mode; it does not restore flat CSR lists in the timestep.

The master APR plan originally assigned all architecture choices to one broad
ADR-0007. This focused plan proposes a cleaner decision graph: ADR-0007 will
cover only the multilevel NNPS after its kill tests. Particle-pool and
split/merge choices receive later ADRs once the P0 stencil mismatch is resolved.
Approval of this plan approves that narrower ADR responsibility; it does not
select production split weights.

P0's Vacondio/PySPH stencil discrepancy does not block synthetic multilevel
neighbor-search work. The NNPS accepts configurable level boundaries and uses
only `x/y/z/h`; no daughter stencil is embedded in this checkpoint.

## Approach

### 1. Freeze the multilevel contract with CPU oracles

- Define discrete levels by configurable `h_ref`, `level_ratio`, and
  `nlevels`. Initial tests use ratios 2 and 1.2; no production default is
  claimed until later APR evidence.
- Assign each source particle to exactly one level. Each level records a
  conservative upper support bound at least as large as every assigned
  `radius_scale*h_j`; particles outside configured bounds fail loudly rather
  than being silently clipped.
- Preserve the existing symmetric pair contract exactly:

  ```text
  rij^2 < (radius_scale*h_i)^2 OR
  rij^2 < (radius_scale*h_j)^2
  ```

- Add deterministic NumPy/brute-force fixtures before GPU traversal:
  single-level, four-level `h_max/h_min=16`, gradual 1.2 ratio, empty levels,
  level-boundary values, clustered refinement, multiple source/destination
  arrays, and particles near spatial bounds.
- Record current uniform-grid accepted and candidate counts for the clustered
  fixture. Accepted sets must stay identical; candidate counts are the
  optimization target.

### 2. Add an explicit multilevel NNPS prototype

- Add `MultilevelGridWarpNNPS` alongside `UniformGridWarpNNPS`; do not alter the
  existing class's defaults or public behavior.
- Compute particle level, per-level counts, and per-level AABB reductions on
  the GPU. Reading `O(number_of_levels)` scalar metadata to size dense level
  grids is permitted in this checkpoint; reading per-particle `x/y/z/h` is not.
- Give each populated level its own origin, dimensions, and cell size based on
  that level's conservative support bound. This avoids allocating a fine grid
  over the spatial extent occupied only by coarse particles.
- Flatten per-level cell arrays into device storage with metadata arrays:
  level cell offsets, particle offsets/counts, origins, cell sizes,
  `(nx,ny,nz)`, and maximum source support. Keep source particle indices in
  original ParticleArray indexing so equation arrays require no remap.
- Reuse the existing GPU count -> exclusive scan -> scatter pattern. Every
  particle appears in exactly one level, preventing duplicate pair visits.
- Cache one multilevel structure per source array per `update()`, matching the
  current source-array ownership model.
- Add candidate/accepted-pair counters behind an explicit diagnostics flag so
  validation can explain performance without taxing production launches.

### 3. Generate an additive multilevel equation loop

- Add `neighbor_mode='multilevel'` to `warp_codegen.py` as a new cache-key and
  source-generation branch. Existing `'flat'` and `'grid'` emitted source must
  remain byte-identical.
- Pass flattened level metadata/device cell arrays through a sibling launcher
  such as `_multilevel_grid_launch_args`; do not overload the existing scalar
  `_grid_launch_args` contract.
- For each destination particle and source level, compute the query radius as
  `max(radius_scale*h_i, level_max_source_support)`. Convert that radius to the
  necessary cell-index range for that level, then apply the exact pairwise
  symmetric cutoff before equation snippets execute.
- Support 1D/2D/3D and cross-array traversal. The production target and hard
  acceptance fixture are 3D.
- Route `_run_equation_group` and `compute_wcsph_adaptive_timestep` through the
  new mode without changing any existing caller's default.
- Multilevel periodic traversal is explicitly rejected with a clear error in
  this checkpoint. Correct per-level periodic tiling is deferred; silently
  using the non-periodic walk is forbidden.

### 4. Add a device neighbor-cache oracle, not a runtime dependency

- Add a multilevel equivalent of `build_neighbor_cache_gpu()` for tests and
  diagnostics. It may perform the existing small lengths/total-size readback to
  allocate the packed neighbor output.
- Use that cache only to compare complete neighbor index sets with brute force.
  Generated WCSPH kernels continue to walk level cell lists directly and do
  not materialize CSR neighbors each stage.

### 5. Validate equations, transfer behavior, and scaling

- Compare full per-particle neighbor sets against `BruteForceWarpNNPS` for all
  fixtures in fp32 and fp64.
- Compare multilevel versus flat/uniform-grid outputs for summation density,
  fused pressure/viscosity/continuity, and adaptive CFL factors.
- Instrument or monkeypatch device-array host access so steady
  `update(push=False)` fails the test if it calls `.get()`/`.numpy()` for
  particle `x/y/z/h`. The permitted metadata transfer is separately counted.
- Verify update after device-coordinate motion, empty arrays/levels, repeated
  updates, and multiple source arrays.
- Benchmark warm grid build and one representative fused consumer separately.
  Report candidate pairs, accepted pairs, metadata bytes read back, build time,
  kernel time, and peak device memory.

## Acceptance criteria

### Correctness

- Exact neighbor-index set parity with brute force for every destination in all
  deterministic fixtures, including four levels spanning `h_max/h_min=16`.
- No duplicate source index is visited for a destination.
- fp32/fp64 SPH outputs match the flat/brute oracle at dtype-derived tolerance;
  adaptive timestep matches after scalar rounding.
- Existing uniform-grid, periodic, generated-source golden, elliptical-drop,
  fixed-wall dam-break, and rigid-coupling tests remain unchanged.

### Device residency

- No per-particle coordinate or smoothing-length host readback during warm
  `update(push=False)` or generated equation traversal.
- Host metadata transfer is bounded by `O(nlevels * narrays)`, is explicitly
  measured, and contains no particle property vectors.
- Level cell lists, scans, particle indices, and traversal remain on the GPU.

### Scaling

- On a localized four-level synthetic fixture, multilevel candidate-pair count
  is at least 4x lower than the current global-`hmax` uniform grid while
  accepted neighbors remain identical.
- A single-level multilevel fixture has no more than 20% warm kernel-time
  overhead versus the current uniform grid. If it misses, single-level callers
  continue using the existing class and the result is disclosed rather than
  hidden.
- At least one mixed-level case shows lower measured build-plus-consumer time,
  not merely fewer candidates. No general speedup claim is made from one case.

### Decision gate

- ADR-0007 is created only after the exact-set and candidate-scaling kill tests
  pass. It records the level representation, device metadata contract,
  cross-level traversal, permitted scalar readback, and periodic deferral.
- If exact traversal requires unbounded work or dense level grids consume more
  memory than the saved particle state on representative cases, stop and
  compare sparse hash/sort alternatives before accepting the ADR.

## Files expected to change

- `pysph/base/warp_nnps.py` - additive multilevel class, level assignment,
  GPU reductions/count/scan/scatter, debug cache, and diagnostics.
- `pysph/base/warp_codegen.py` - additive `'multilevel'` generated traversal
  and cache key; old modes byte-identical.
- `pysph/base/warp_sph.py` - multilevel launch arguments and explicit routing
  for generated groups/adaptive timestep.
- `pysph/base/tests/test_warp_nnps.py` - exact set, lifecycle, device-residency,
  and candidate-count tests.
- `pysph/base/tests/test_warp_codegen.py` - multilevel source/signature tests and
  existing-source golden guards.
- `pysph/base/tests/test_warp_sph.py` - equation and timestep parity.
- New P1 experiment artifacts, ADR-0007, aspect/current/session/review/closeout
  memory as required by the operating contract.

All host files match the existing `pysph/base/warp_*.py` and
`pysph/base/tests/test_warp_*.py` boundary. No generic NNPS Cython API or public
GPU export is touched.

## Tests / validation

- Fast kill tests first: level assignment, metadata, exact 2D/3D neighbor sets,
  and candidate counts.
- Focused commands:

  ```text
  python -m pytest -q pysph/base/tests/test_warp_nnps.py
  python -m pytest -q pysph/base/tests/test_warp_codegen.py
  python -m pytest -q pysph/base/tests/test_warp_sph.py
  ```

- Final exact-tree `test_warp_*` suite after warm JIT cache.
- Re-run the 2D generated-source golden and representative no-obstacle,
  obstacle, and rigid-driver smoke tests.
- Record all performance runs in a new P1 experiment with GPU, driver, Warp
  version, commit, precision, particle/level distribution, commands, repeated
  warm timings, candidate counts, accepted counts, memory, and correctness
  hashes.
- `validate-memory.py`, decision-graph regeneration after ADR-0007,
  `git diff --check`, and a prototype-owner review before commit.

## Risks

- Cross-level omissions are silent physics errors; exact brute-force set parity
  is a non-negotiable gate.
- A coarse destination querying a dense fine level may legitimately span many
  fine cells. The hierarchy removes artificial work but cannot remove real
  cross-scale neighbors.
- Per-level dense grids may still be wasteful for disconnected fine regions;
  sparse key/sort storage may be required after the kill test.
- Runtime loops over levels can reduce GPU occupancy or inflate generated Warp
  code. Measure one fused consumer before expanding scope.
- GPU AABB reductions and metadata readback introduce synchronization. The
  transfer is bounded but must be timed separately.
- Adding a new generated structural mode causes new cold JIT compilation. It
  must not perturb existing cached source.
- The current P0 stencil mismatch means level ratios are provisional. The class
  therefore remains configurable and prototype-only.

## Out of scope

- Runtime particle splitting, merging, allocation, free lists, or compaction.
- Variable-`h` physics corrections and particle shifting.
- Geometry- or solution-driven target-level assignment.
- Periodic multilevel domains.
- Local/multirate timestepping.
- Multi-GPU/MPI decomposition.
- Changes to generic PySPH NNPS APIs, Cython ABI, or shipped applications.
- Selecting or publishing production APR split weights.

## Estimated effort

Two to three substantial implementation sessions, approximately 500--800 LOC
across backend/tests plus experiment and decision artifacts. The exact-set and
candidate-count fixtures are the first kill gate; generated SPH integration
does not start until they pass.

## Approval

- [x] Plan posted in chat
- Approved by: @____ at 2026-07-06T__:__:__ CEST
- Approval, verbatim quote:
  > {{exact user message}}
