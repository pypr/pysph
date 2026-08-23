---
type: plan
id: 2026-06-15_nnps-solver-agnostic-spec-and-warp-prototype
author: codex
agent: codex
created: 2026-06-15T09:30:00 CET
status: approved
aspects: [gpu-nnps, particle-memory, warp-backend, validation-benchmarks]
host_files: [pysph/base/nnps_base.pyx, pysph/base/gpu_nnps_base.pyx, pysph/base/gpu_nnps_base.pxd, pysph/base/gpu_nnps.py, pysph/solver/application.py]
within_boundary: false
---

# Plan: NNPS Solver-Agnostic Spec And Warp Prototype

## Trigger

After the Warp ParticleArray mirror checkpoint, the next high-level migration
step is neighbor search. The codebase understanding document identifies NNPS as
the bridge between particle storage and generated equation loops.

## Objective

Define the solver-agnostic NNPS contract and prepare a Warp NNPS prototype that
can consume Warp-backed ParticleArray data without requiring full solver or SPH
equation migration.

## Proposed Work

1. Capture NNPS/domain/update/query/cache semantics in
   `spec/nnps/`.
2. Add CPU-vs-Warp NNPS correctness experiments.
3. Implement a minimal Warp brute-force NNPS for correctness.
4. Implement a Warp cell-list NNPS for performance relevance.
5. Add explicit host integration only after the direct NNPS API passes.

## Progress

- User approved beginning NNPS implementation with: "ok. lets begin with the
  NNPS implementation with warp".
- Added `pysph/base/warp_nnps.py` with `BruteForceWarpNNPS`.
- Added focused correctness tests in `pysph/base/tests/test_warp_nnps.py`.
- Added experiment packet
  `experiments/2026-06-15_warp-nnps-bruteforce-baseline/`.
- Smoke benchmark confirms CPU and Warp average neighbor counts match, while
  Warp brute force is slower due per-query kernel launch and readback.
- Added cached flat-neighbor-list mode to reduce per-query launch/readback
  overhead.
- Added `UniformGridWarpNNPS`, which builds source cell lists on the device and
  queries adjacent cells.
- Focused tests now cover the grid path in 1D/2D/3D, multiple arrays, variable
  smoothing length, and update-after-mutation cases.

## Initial Success Criteria

- Spec identifies query contract, update timeline, domain/ghost semantics,
  cache behavior, parallel boundary, and verification fixtures.
- First experiment compares CPU NNPS and Warp NNPS neighbor sets.
- Warp NNPS benchmark reports update, query, cache, and readback time
  separately.

## Out Of Scope

- Migrating generated SPH equation kernels.
- Replacing MPI/Zoltan partitioning.
- Supporting every CPU NNPS variant immediately.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-15T10:05:00 CET
- Approval, verbatim quote:
  > ok. lets begin with the NNPS implementation with warp
