---
aspect: warp-backend
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T07:19:08 CET
status: active
---

# Aspect: warp-backend

## What this aspect covers

NVIDIA Warp API choices, kernel model, memory layout assumptions, and how Warp could map onto PySPH's existing GPU abstractions.

## Current understanding

Initial scaffolding - to be filled in the first working session on this aspect.

## Key sub-topics

- Warp version/API surface - Confirm with team.
- Kernel launch model - Confirm with team.
- Compatibility with existing PySPH GPU pathways - Confirm with team.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: warp-backend` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - device data ownership.
- Influences: `gpu-nnps` - backend-specific neighbor kernels.
