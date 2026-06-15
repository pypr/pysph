# Host Integration Contract

## Construction

An NNPS object is constructed from:

- dimension
- particle-array list
- radius scale
- ghost layers
- optional domain manager
- cache flag
- sorted-gid flag
- backend-specific options

The constructor may call `domain.update()` and `update()` immediately, as
existing concrete NNPS classes do.

## Application Selection

Application setup currently selects GPU NNPS when OpenCL or CUDA flags are set.
For Warp, integration should make backend selection explicit and avoid
pretending to be the existing CUDA/Compyle backend.

Candidate host surfaces:

- `--backend warp`
- `--nnps warp_ll`
- `--nnps warp_bruteforce`
- `backend="warp"` in direct Python construction

## Output And Serialization

NNPS state is not solver output. Output files serialize particle arrays and
solver metadata, not cell structures or neighbor caches.

Warp NNPS therefore only needs to leave particle arrays in a host-readable
state when output or post-processing asks for them.

## Spatial Reordering

`spatially_order_particles(pa_index)` obtains an index permutation and aligns
particle properties. Warp implementations should reuse the ParticleArray Warp
alignment semantics and update or invalidate all NNPS caches afterward.
