---
aspect: validation-benchmarks
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T07:19:08 CET
status: active
---

# Aspect: validation-benchmarks

## What this aspect covers

Baselines, timings, correctness checks, acceptance thresholds, experiment handoff, and validation evidence for fast particle dynamics.

## Current understanding

Initial scaffolding - to be filled in the first working session on this aspect.

## Key sub-topics

- Baseline selection.
- Hardware/runtime recording.
- Correctness tolerance and performance thresholds.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: validation-benchmarks` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `gpu-nnps`, `particle-memory`, and `warp-backend`.
- Influences: success criteria and review evidence.
