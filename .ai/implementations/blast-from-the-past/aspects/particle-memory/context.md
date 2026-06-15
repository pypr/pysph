---
aspect: particle-memory
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T07:19:08 CET
status: active
---

# Aspect: particle-memory

## What this aspect covers

ParticleArray/device data ownership, transfer semantics, dtype/precision, and compatibility with existing PySPH device helpers.

## Current understanding

Initial scaffolding - to be filled in the first working session on this aspect.

## Key sub-topics

- ParticleArray property ownership.
- Device helper compatibility.
- Float/double precision choices.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: particle-memory` and `scope: global`.

## Cross-aspect dependencies

- Influences: `warp-backend`, `gpu-nnps`, and `validation-benchmarks`.
