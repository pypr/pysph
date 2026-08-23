# Boundary And Ghost Semantics

## Tags

Particle tags affect which particles are local, remote, or ghost. NNPS itself
queries over the particle arrays it receives; domain and parallel managers are
responsible for making the right local/remote/ghost particles present before
NNPS update.

## Periodic Domains

For periodic axes, the domain manager creates ghost particles translated by the
domain period. NNPS then treats those ghost particles as ordinary source
particles during geometric queries.

Required Warp behavior:

- old periodic ghosts are removed before new ghosts are created;
- copied ghost properties match the domain manager's copy-property selection;
- neighbor results include ghost source indices when ghosts are present;
- real-particle count remains consistent after ghost insertion/removal.

## Mirror Domains

The current backend selector warns that mirrored boundaries are unsupported by
existing GPU domain manager paths and falls back to CPU domain management.

The first Warp NNPS may defer mirror-domain support, but the integration must
fail clearly or fall back explicitly rather than silently returning incomplete
neighbors.

## Out-Of-Domain Particles

The domain manager is responsible for detecting or tolerating bounds changes.
NNPS should preserve the current warning behavior when domain size grows by a
large factor between updates.
