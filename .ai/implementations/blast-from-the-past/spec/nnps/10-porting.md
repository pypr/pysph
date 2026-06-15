# Porting Plan Seed

## Premises

- ParticleArray now has a Warp device mirror prototype.
- Existing NNPS code expects ParticleArray host properties and Cython wrappers.
- Existing GPU NNPS code is Compyle/OpenCL/CUDA-oriented and uses
  `GPUNeighborCache` to bridge GPU and CPU neighbor access.
- The first Warp migration should be additive and should not remove existing
  CPU or Compyle GPU paths.

## Phase 1: NNPS Spec And Experiment

1. Capture this solver-agnostic NNPS contract.
2. Add CPU-vs-Warp neighbor correctness experiments.
3. Add benchmark scripts that separate update, query, cache, and readback time.

## Phase 2: Warp Brute Force

1. [DONE] Implement a minimal Warp NNPS using ParticleArray Warp arrays.
2. [DONE] Provide `update()`, `set_context()`, and
   `get_nearest_particles()`.
3. [DONE] Keep a host-compatible neighbor readback path for existing tests.
4. [DONE] Validate source/destination array pairs and variable `h`.
5. [DONE] Avoid one kernel launch and one full flag readback per destination
   particle with a cached flat-neighbor-list path.
6. [DONE] Replace brute-force O(N^2) cache construction with a cell-list
   implementation baseline.
7. [DONE] Add a narrow device-resident consumer that uses the grid cache for
   equation-like work.
8. [DONE] Use the device-resident cache for a real SPH equation:
   CubicSpline summation density.
9. [DONE] Add EOS and continuity as the next simple SPH kernels.
10. [DONE] Add inviscid pressure-gradient acceleration.
11. [DONE] Add a minimal one-step WCSPH Euler chain that consumes density, EOS,
   and pressure-gradient outputs on the device.
12. [NEXT] Generalize the device-consumption proof into a reusable
   equation-loop contract and repeated-step NNPS refresh.

## Phase 3: Warp Cell List

1. [DONE] Compute cell ids on the device.
2. [DONE] Build per-cell counts and offsets.
3. [DONE] Scatter particle ids into a flat cell-particle array.
4. [DONE] Query adjacent cells on the device.
5. [DONE] Compare against brute-force Warp and CPU baselines.
6. [DONE] Reduce host readback for bulk cache construction and a neighbor-sum
   consumer.
7. [NEXT] Tune cell-list performance and cache reuse across multiple equation
   consumers.

## Phase 4: Solver Integration

1. Add explicit Application/CLI selection for Warp NNPS.
2. Run a small example with Warp ParticleArray plus Warp NNPS.
3. Keep equation evaluation on the existing backend until the neighbor contract
   is stable.

## Phase 5: Equation Kernel Consumption

1. [DONE] Prove generated-equation-like kernels can consume cached neighbor
   lists directly with `compute_neighbor_sum()`.
2. [DONE] Port and benchmark standard `SummationDensity` as the first real SPH
   equation kernel.
3. [DONE] Port and benchmark `IsothermalEOS` plus `ContinuityEquation`.
4. [DONE] Port and benchmark inviscid pressure-gradient acceleration.
5. [DONE] Add a minimal Warp-aware acceleration-to-position step.
6. Add a device-aware repeated-step NNPS refresh.
7. Benchmark end-to-end solver steps.

## Risks

- Existing Cython equation paths expect `UIntArray` neighbor results.
- Host readback can hide GPU performance wins.
- Periodic and mirror ghost behavior can create correctness mismatches even
  when raw geometric queries are correct.
- Parallel remote-particle exchange may reorder arrays in ways that require
  sorted-gid comparisons.
- Repeated GPU stepping needs NNPS update semantics that do not push stale host
  positions over device-updated positions.
