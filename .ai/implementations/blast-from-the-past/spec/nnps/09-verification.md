# Verification

## Correctness Fixtures

Required deterministic fixtures:

- 1D evenly spaced particles with fixed `h`
- 2D grid with fixed `h`
- 3D small cloud with fixed `h`
- variable smoothing length where gather radius and scatter radius differ
- multiple particle arrays with source/destination indices crossed
- empty source array
- empty destination array
- duplicate positions
- sorted-gid and unsorted neighbor ordering
- periodic boundary ghost inclusion
- post-add/remove ParticleArray mutation followed by NNPS update

## Baseline Comparisons

For most fixtures, compare Warp neighbor sets to CPU linked-list or brute-force
NNPS. Compare order only when `sort_gids=True`.

For cached paths, verify:

- first query materializes the cache;
- repeated query returns the same neighbors;
- update invalidates the cache;
- mutation followed by update produces the new expected neighbors.

## Performance Metrics

Record:

- particle count
- dimension
- average neighbor count
- smoothing-length mode
- backend
- update time
- query time for all destination particles
- cache build time
- device-to-host readback time, if any
- memory footprint for neighbor lengths, starts, and flat list

## Success Criteria

The first Warp NNPS experiment succeeds when:

- serial correctness fixtures match CPU baselines;
- all results stay device-resident until an explicit host query/readback;
- benchmark output identifies whether time is spent in bounds, binning, query,
  cache construction, or readback;
- an agreed particle-count threshold shows a measurable speedup over CPU for
  at least one realistic update/query workload.
