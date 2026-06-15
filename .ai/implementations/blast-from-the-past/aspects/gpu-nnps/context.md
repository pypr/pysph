---
aspect: gpu-nnps
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T12:30:00 CET
status: active
---

# Aspect: gpu-nnps

## What this aspect covers

Neighbor-search design and performance around `GPUNNPS`, GPU neighbor caches, and GPU neighbor-list construction.

## Current understanding

NNPS is the next migration layer after ParticleArray. It owns the geometric
neighbor-query contract between particle storage and generated equation loops.

The solver-agnostic NNPS spec now lives under
`.ai/implementations/blast-from-the-past/spec/nnps/`. It covers domain/bounds
state, update timeline, source/destination query semantics, pairwise
smoothing-length inclusion, cache behavior, boundary ghosts, MPI/Zoltan
boundaries, host selection, variant ordering, verification fixtures, and a Warp
porting plan seed.

Existing PySPH surfaces observed for this spec:

- `DomainManager` selects CPU/GPU domain management and owns periodic/mirror
  settings.
- `NNPSBase` owns particle arrays, wrappers, radius scale, caches, and query
  context.
- CPU `NNPS.update()` computes bounds, refreshes structure storage, bins each
  particle array, and refreshes caches.
- Existing `GPUNNPS` uses Compyle/OpenCL/CUDA helpers and `GPUNeighborCache`;
  it is not Warp-native.
- Application setup currently chooses `OctreeGPUNNPS` or `ZOrderGPUNNPS` for
  existing OpenCL/CUDA modes.

First Warp implementation:

- `pysph/base/warp_nnps.py` defines `BruteForceWarpNNPS`.
- It uses Warp kernels for the pairwise distance test and returns neighbors
  through `UIntArray`.
- It supports source/destination array pairs, 1D/2D/3D coordinate selection,
  variable source/destination `h`, sorted-gid output, and update after host
  ParticleArray mutation.
- It supports an uncached per-query flags path and a cached flat-neighbor-list
  path.
- It is intentionally not the final performance target: the cached path avoids
  per-destination launch/readback but remains brute-force O(N^2).

Uniform-grid implementation:

- `UniformGridWarpNNPS` builds per-source device-side cell ids, cell counts,
  exclusive-scan cell starts, and flat cell-particle arrays.
- Neighbor caches are built by scanning adjacent cells and applying the same
  pairwise `h_i`/`h_j` inclusion rule.
- The first grid path supports 1D/2D/3D, multiple particle arrays, variable
  `h`, and update after mutation in focused tests.
- It still materializes host-side neighbor arrays for the existing `UIntArray`
  query contract; equation-kernel integration should avoid that readback.
- `warp_grid_device` benchmarks bulk device neighbor-cache construction without
  the per-particle `UIntArray` query loop. At 1,000,000 particles on
  PrediQT-02, it measured `88.288x` CPU speed while matching average neighbor
  count.

Device-consumption proof:

- `UniformGridWarpNNPS.compute_neighbor_sum(src_index, dst_index, prop)` builds
  the device-resident neighbor cache and runs a Warp kernel that sums a scalar
  source property over neighbors for each destination particle.
- This is intentionally a narrow equation-like consumer, not the final solver
  loop. It proves the cache can feed useful GPU work before any host
  `UIntArray` materialization.
- At 1,000,000 particles on PrediQT-02, `warp_grid_reduce` measured `145.583x`
  CPU speed for a neighbor mass sum on Intel(R) Core(TM) Ultra 7 155H versus
  NVIDIA GeForce RTX 4060 Laptop GPU. The average neighbor sum matched to the
  reported precision (`25.568`); the aggregate checksum differed by `6` over
  roughly `25.6M` contributions.

First SPH equation proof:

- `pysph/base/warp_sph.py` defines `compute_summation_density()` and Warp
  CubicSpline density kernels for float32/float64.
- The kernel mirrors PySPH `SummationDensity` with
  `HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])` and
  `rho_i = sum_j m_j * W(XIJ, RIJ, HIJ)`.
- Focused tests compare Warp density values against a CPU `CubicSpline`
  reference in 2D and cross-array 3D and verify `rho` can be pulled back to the
  host ParticleArray.
- At 10,000,000 particles on PrediQT-02, `warp_grid_density` measured
  `69.084x` CPU/Cython speed versus PySPH `SPHEvaluator` with
  `SummationDensity`, `CubicSpline(dim=2)`, and `LinkedListNNPS`.

## Key sub-topics

- Existing `GPUNeighborCache` behavior.
- Existing brute-force, Z-order, stratified SFC, and octree GPU NNPS surfaces.
- Correctness and performance baselines.
- Warp brute-force correctness baseline.
- Warp cell-list performance prototype.
- Cached flat neighbor list generation.
- Optimize uniform-grid/cell-list structure.
- Device-resident equation-kernel consumption of grid neighbor lists.
- Reusable Warp equation-loop contract.
- Warp SPH equation kernels.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: gpu-nnps` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - particle/device arrays.
- Influences: `validation-benchmarks` - neighbor-search benchmark cases.
