---
aspect: gpu-nnps
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T09:30:00 CET
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

## Key sub-topics

- Existing `GPUNeighborCache` behavior.
- Existing brute-force, Z-order, stratified SFC, and octree GPU NNPS surfaces.
- Correctness and performance baselines.
- Warp brute-force correctness baseline.
- Warp cell-list performance prototype.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: gpu-nnps` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - particle/device arrays.
- Influences: `validation-benchmarks` - neighbor-search benchmark cases.
