# Open Questions

- [UNKNOWN] Should the first Warp NNPS target be a brute-force baseline for
  correctness, or a uniform-grid/cell-list implementation for immediate
  performance relevance?
- [UNKNOWN] Should `warp` appear as a new `--nnps` value, a new global backend
  option, or both?
- [UNKNOWN] How much of the existing `GPUNeighborCache` CPU readback protocol
  must be preserved for generated equation code during the first integration?
- [UNKNOWN] Is mirrored-boundary support required in the first Warp NNPS, or can
  it follow periodic and non-periodic domains?
- [UNKNOWN] What particle counts and neighbor densities define "blazing fast"
  for NNPS update/query benchmarks on `prediqt-02`?
