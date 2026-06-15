# Data Structures

## Required Particle Properties

At minimum, NNPS needs:

- `x`, `y`, `z`: coordinates
- `h`: smoothing length
- `tag`: local, remote, ghost, or other particle status
- `gid`: global id, used when sorted neighbor order is requested

Particle arrays may contain many additional properties, but NNPS must not
depend on them for geometric neighbor selection.

## Particle Array Wrappers

The CPU path uses `NNPSParticleArrayWrapper` to access typed property arrays and
to remove tagged particles through the owning `ParticleArray`.

A Warp NNPS may either:

- reuse the wrapper for host compatibility and read device arrays from
  `pa.gpu`, or
- introduce a Warp-specific wrapper exposing the same conceptual fields.

The second option is preferable once NNPS stops round-tripping through host
arrays.

## Structure Storage

Backend-neutral storage concepts:

- per-array particle count
- coordinate bounds
- cell size
- cell id per particle
- cell occupancy or head/next links
- optional spatially ordered index permutation
- optional neighbor cache lengths
- optional neighbor cache start offsets
- optional flat neighbor index array

The CPU linked-list implementation stores `head` per cell and `next` per
particle. Existing GPU implementations store neighbor lengths, prefix-summed
start indices, and flat neighbor lists for cached GPU access.

## Invalid Sentinel

The CPU linked-list implementation uses `UINT_MAX` as an invalid particle/cell
link sentinel. Warp kernels should use a documented unsigned sentinel if they
mirror linked-list storage.
