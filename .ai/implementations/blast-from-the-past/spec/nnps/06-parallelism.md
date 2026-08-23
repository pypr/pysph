# Parallelism

## MPI/Zoltan Boundary

MPI/Zoltan partitioning remains outside NNPS. Before NNPS update, the parallel
manager may:

- remove stale remote particles
- migrate exported local particles
- import remote particles
- compute ghost/remote particles required by neighbor overlap
- update local and remote cell maps

NNPS assumes that the particle arrays passed to it already contain the particles
needed for local computation.

## Local Versus Remote Queries

Neighbor results are source-local indices into the current process's particle
arrays. They do not identify MPI ranks.

For parallel comparisons, `sort_gids=True` helps serial and parallel outputs
match by global id rather than by process-local insertion order.

## Reduction And Timestep

NNPS does not own timestep reduction. Adaptive timestep minimum reductions
belong to solver/parallel manager code.

## Warp/MPI First Cut

The first Warp NNPS should be validated in serial. A follow-up parallel
experiment should verify:

- remote particles survive device sync and NNPS update;
- neighbor lists over local+remote arrays match CPU NNPS;
- Zoltan load balance followed by Warp NNPS update is deterministic enough for
  existing parallel comparison tolerances.
