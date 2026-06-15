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

1. Implement a minimal Warp NNPS using ParticleArray Warp arrays.
2. Provide `update()`, `set_context()`, and `get_nearest_particles()`.
3. Keep a host-compatible neighbor readback path for existing tests.
4. Validate source/destination array pairs and variable `h`.

## Phase 3: Warp Cell List

1. Compute cell ids on the device.
2. Build per-cell counts and offsets.
3. Scatter particle ids into a flat cell-particle array.
4. Query adjacent cells on the device.
5. Compare against brute-force Warp and CPU baselines.

## Phase 4: Solver Integration

1. Add explicit Application/CLI selection for Warp NNPS.
2. Run a small example with Warp ParticleArray plus Warp NNPS.
3. Keep equation evaluation on the existing backend until the neighbor contract
   is stable.

## Phase 5: Equation Kernel Consumption

1. Decide whether generated equation kernels consume cached neighbor lists or
   invoke Warp neighbor-query kernels directly.
2. Add a Warp-aware acceleration-evaluation plan.
3. Benchmark end-to-end solver steps.

## Risks

- Existing Cython equation paths expect `UIntArray` neighbor results.
- Host readback can hide GPU performance wins.
- Periodic and mirror ghost behavior can create correctness mismatches even
  when raw geometric queries are correct.
- Parallel remote-particle exchange may reorder arrays in ways that require
  sorted-gid comparisons.
