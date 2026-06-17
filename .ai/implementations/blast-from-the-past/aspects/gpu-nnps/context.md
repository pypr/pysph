---
aspect: gpu-nnps
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T14:25:00 CET
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

EOS and continuity proof:

- `pysph/base/warp_sph.py` now also defines `compute_isothermal_eos()` and
  `compute_continuity()`.
- EOS mirrors PySPH `IsothermalEOS`: `p = p0 + c0^2*(rho-rho0)`.
- Continuity mirrors PySPH `ContinuityEquation` with `VIJ . DWIJ` and the same
  CubicSpline gradient convention at `HIJ`.
- Focused tests compare EOS and continuity against CPU references in same-array
  2D and cross-array 3D fixtures, including host pullback of `p` and `arho`.
- The EOS+continuity benchmark is capped at 5M particles. At 5M on PrediQT-02,
  `warp_grid_eos_cont` measured `72.583x` CPU/Cython speed versus PySPH
  `SPHEvaluator`.

Pressure-gradient proof:

- `pysph/base/warp_sph.py` now defines `compute_pressure_gradient()` for the
  inviscid pressure-gradient portion of WCSPH momentum.
- The kernel computes
  `a_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad(W_ij)` using the
  same CubicSpline gradient convention at `HIJ`.
- Focused tests compare same-array 2D and cross-array 3D accelerations against
  CPU references, including host pullback of `au`, `av`, and `aw`.
- The pgrad benchmark is capped at 5M particles. At 5M on PrediQT-02,
  `warp_grid_pgrad` measured `38.722x` CPU/Cython speed versus a pure Cython
  pressure-gradient equation.

Repeated-step proof:

- `UniformGridWarpNNPS.update(push=False)` can rebuild bounds, grids, and
  caches from device-resident `x/y/z/h` values without pushing stale host
  ParticleArray coordinates over the device state.
- `pysph/base/warp_sph.py` now has device-side leapfrog kick/drift kernels,
  periodic position wrapping, and a minimal `wc_sph_leapfrog_step()`.
- Focused tests compare the KDK step against CPU reference density, EOS,
  pressure-gradient acceleration, and final state. The Warp SPH/NNPS suite
  passes with `29 passed`.
- Periodic behavior is currently position wrapping only. Minimum-image distance
  and periodic cell lookup remain open for true periodic neighbor interaction.

Artificial-viscosity proof:

- `pysph/base/warp_sph.py` now adds Monaghan-style artificial viscosity through
  the same device-resident `UniformGridWarpNNPS` neighbor cache used by density,
  continuity, and pressure-gradient kernels.
- The viscosity kernel is additive over existing `au/av/aw`, so the pressure
  gradient path can remain the owner of resetting acceleration before optional
  stabilizing terms contribute.
- The current implementation uses per-particle sound speed `cs` when available,
  with a constant-`c0` fallback for callers that have not run Tait EOS.

Tait EOS proof:

- `pysph/base/warp_sph.py` now adds a Warp `TaitEOS` path that writes both
  pressure `p` and sound speed `cs`.
- The WCSPH step helpers keep `eos='isothermal'` as the compatibility default
  and accept `eos='tait'`, `gamma=7.0` for the elliptical-drop path.
- Artificial viscosity now consumes `cs` through the same device-resident
  neighbor cache and computes `cij = 0.5*(d_cs + s_cs)`.

Continuity-density repeated-step proof:

- `wc_sph_leapfrog_step(..., density_mode='continuity')` now routes to a
  PySPH `WCSPHStep`-style PEC path. It saves reference position/velocity/rho
  state on device, computes Tait EOS, pressure-gradient/artificial-viscosity
  acceleration, `ContinuityEquation` density rate, and XSPH correction from the
  device-resident uniform-grid neighbor cache, applies stage1, rebuilds NNPS
  from device positions, recomputes equations, and applies stage2.
- The original KDK summation-density step remains the compatibility/default
  path. The continuity path is the one used for PySPH Application parity in the
  resolved elliptical-drop comparison.

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
- Tiny Euler/PEC-style integrator loop.
- Device-authoritative NNPS refresh after position updates.
- Minimal KDK leapfrog step and periodic position wrapping.
- Additive artificial-viscosity momentum term.
- Tait EOS and per-particle sound-speed path.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: gpu-nnps` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - particle/device arrays.
- Influences: `validation-benchmarks` - neighbor-search benchmark cases.
