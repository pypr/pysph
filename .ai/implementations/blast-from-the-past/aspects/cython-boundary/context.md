---
aspect: cython-boundary
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T08:34:00 CET
status: active
---

# Aspect: cython-boundary

## What this aspect covers

What remains in `.pxd/.pyx`, what can be wrapped or bypassed, and how to preserve Cython ABI/API expectations while experimenting with Warp.

## Current understanding

ParticleArray is a Cython extension type declared in `pysph/base/particle_array.pxd`, and existing code can rely on cpdef/cdef methods plus `BaseArray` returns from `get_carray()`. The first Warp migration should preserve the existing ParticleArray public API and host carray compatibility unless a separate boundary amendment approves broader ABI/API changes.

## Key sub-topics

- `.pxd` declaration compatibility.
- Cython extension build constraints.
- Host boundary amendments.
- Compatibility with `BaseArray`/NumPy readback paths.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: cython-boundary` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `host-integration` - build/test constraints.
- Influences: all host code changes.
