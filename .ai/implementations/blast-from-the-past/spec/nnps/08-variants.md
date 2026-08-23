# Variants

## CPU Variants

Observed CPU-facing NNPS variants include:

- box sort
- linked list
- spatial hash
- extended spatial hash
- cell indexing
- z-order
- stratified SFC
- octree
- compressed octree
- brute-force fallback behavior

## Existing GPU Variants

Existing GPU exports include:

- `BruteForceNNPS`
- `ZOrderGPUNNPS`
- `StratifiedSFCGPUNNPS`
- `OctreeGPUNNPS`
- `GPUDomainManager`
- `GPUNeighborCache`

These are Compyle/OpenCL/CUDA-oriented, not Warp-native.

## Recommended Warp Variant Order

1. Warp brute force: simplest correctness oracle and device-array plumbing.
2. Warp uniform grid or linked-cell list: first performance-relevant structure.
3. Warp cached flat neighbor lists: bridge to generated equation kernels.
4. Warp spatial reordering: improves memory locality and solver-loop cost.
5. Warp octree or SFC variants: only after simpler structures establish wins.

## Fixed-H Optimization

When smoothing lengths are fixed, cell size and some structure allocations can
be reused across updates. This is an optimization only; the correctness
contract remains the pairwise inclusion rule.
