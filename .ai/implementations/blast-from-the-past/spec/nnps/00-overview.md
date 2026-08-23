# NNPS Solver-Agnostic Specification

## Scope

This specification describes the nearest-neighbor particle search subsystem in
solver-agnostic terms. It is intended to guide an NVIDIA Warp migration without
assuming any one SPH formulation.

NNPS sits between particle storage and equation evaluation:

1. Particle arrays provide positions, smoothing lengths, tags, gids, and array
   ordering.
2. Domain management updates bounds, cell size, and periodic or mirror ghost
   particles.
3. NNPS builds an acceleration structure over source particle arrays.
4. Equation loops query neighbors for each destination particle.

## Observed Entry Points

- `DomainManager` wraps CPU or GPU domain manager selection.
- `NNPSBase` owns particle arrays, particle-array wrappers, radius scale, cache
  state, and source/destination query context.
- `NNPS.update()` refreshes bounds, structure storage, particle binning, and
  optional neighbor caches.
- `get_nearest_particles(src_index, dst_index, d_idx, nbrs)` returns neighbors
  for one destination particle.
- Existing GPU paths expose `GPUNNPS`, `GPUNeighborCache`,
  `get_nearest_particles_gpu()`, `find_neighbor_lengths()`, and
  `find_nearest_neighbors_gpu()`.

## Backend-Neutral Contract

An NNPS implementation must answer this question:

> Given a destination particle index and a source particle array, which source
> particles lie within the pairwise interaction radius implied by destination
> and source smoothing lengths?

The inclusion rule used by the CPU baseline is:

```text
distance(i, j) < radius_scale * h_i
or
distance(i, j) < radius_scale * h_j
```

where `i` is the destination particle and `j` is a source particle.

## Non-Goals For This Spec

- It does not prescribe SPH equations, kernels, or integrator stages.
- It does not replace MPI/Zoltan partitioning.
- It does not require one acceleration structure.
- It does not decide whether generated equation kernels consume compressed
  neighbor lists or invoke query kernels directly.
