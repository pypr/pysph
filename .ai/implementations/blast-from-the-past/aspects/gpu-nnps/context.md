---
aspect: gpu-nnps
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T07:19:08 CET
status: active
---

# Aspect: gpu-nnps

## What this aspect covers

Neighbor-search design and performance around `GPUNNPS`, GPU neighbor caches, and GPU neighbor-list construction.

## Current understanding

Initial scaffolding - to be filled in the first working session on this aspect.

## Key sub-topics

- Existing `GPUNeighborCache` behavior.
- Existing brute-force, Z-order, stratified SFC, and octree GPU NNPS surfaces.
- Correctness and performance baselines.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: gpu-nnps` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - particle/device arrays.
- Influences: `validation-benchmarks` - neighbor-search benchmark cases.
