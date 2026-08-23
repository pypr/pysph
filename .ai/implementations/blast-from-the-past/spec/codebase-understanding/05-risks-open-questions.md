# Risks And Open Questions

## Build And Environment Risks

- Cython extension compilation depends on local compiler availability.
- MPI/Zoltan support depends on matching headers, libraries, and Python
  packages.
- Generated-code behavior can depend on `~/.compyle/config.py`.
- Existing CI targets Python 3.11 and 3.12, while the PQT environment currently
  exercises newer Python behavior.

## Runtime Risks

- GPU execution depends on local device availability and backend-specific kernel
  generation.
- MPI execution requires both MPI and Zoltan.
- Particle property names and equation signatures are compile/setup-time
  contracts.
- Output paths expect host-readable particle arrays.

## Migration Risks

- A Warp ParticleArray mirror can appear correct while NNPS still forces
  host/device synchronization.
- A Warp NNPS can be geometrically correct while boundary ghost semantics are
  wrong.
- Sorted neighbor comparisons may differ by local index unless `gid` behavior is
  intentionally handled.
- Parallel exchange can reorder arrays in ways that invalidate cache or spatial
  ordering assumptions.

## Open Questions To Carry Forward

- What is the first measurable "blazing fast" threshold for NNPS on
  `prediqt-02`?
- Should Warp NNPS first expose a brute-force correctness backend or go straight
  to a cell-list implementation?
- Should Warp backend selection be global, NNPS-specific, ParticleArray-specific,
  or a combination?
- Which end-to-end example is the first acceptance case after NNPS integration:
  `elliptical_drop`, `cavity`, or a smaller synthetic application?
