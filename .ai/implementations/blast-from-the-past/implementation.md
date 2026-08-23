---
implementation: blast-from-the-past
host_project: PySPH
created: 2026-06-15T07:19:08 CET
owner: @kunalpuri-prediqt
status: active
---

# blast-from-the-past

## Summary

Nostalgia and exploration with NVIDIA Warp as a path toward PySPH on GPUs.

## Goal

PySPH on GPUs.

## Non-goals

Do not reinvent SPH formulations; focus on GPU execution and integration.

## Success criteria

Blazing fast particle dynamics. Confirm with team: define concrete timing, throughput, hardware, and correctness thresholds before first benchmark claims.

## Integration boundary

The set of host files, modules, and interfaces this implementation interacts with. Anything outside this list is out of scope unless approved through a boundary amendment.

- `pysph/**/*.pxd` - Cython declarations and ABI/public extension surfaces that may constrain Warp integration.
- `pysph/**/*.pyx` - Cython implementation files that may need wrapping, comparison, or future implementation changes.
- `pysph/base/gpu_nnps.py` - GPU NNPS export surface read during discovery because it re-exports the active GPU NNPS classes.
- `pysph/base/warp_*.py` - Python Warp prototype helpers for ParticleArray,
  NNPS, and SPH equation/integrator checkpoints.
- `pysph/base/tests/test_warp_*.py` - focused tests for the Python Warp
  prototype helpers.
- `CODEBASE_UNDERSTANDING.md` - repository architecture snapshot retained as a
  host-level reference at the implementation owner's request.

## Boundary amendments

- 2026-06-16 - Added Python Warp prototype files and focused Warp tests to the
  active implementation boundary for the repeated-step leapfrog checkpoint.
- 2026-06-21 - Added the owner-provided root `CODEBASE_UNDERSTANDING.md`
  architecture snapshot so it can be versioned with the curated implementation
  spec rather than left as an unexplained untracked file.

## Aspects

- `warp-backend` - Tracks NVIDIA Warp API choices, kernel model, memory layout assumptions, and how Warp maps onto PySPH GPU abstractions.
- `gpu-nnps` - Tracks neighbor-search design and performance around `GPUNNPS`, caches, and GPU neighbor lists.
- `particle-memory` - Tracks ParticleArray/device data ownership, transfers, dtype/precision, and compatibility with existing device helpers.
- `cython-boundary` - Tracks what remains in `.pxd/.pyx`, what can be wrapped or bypassed, and how to preserve ABI/API expectations.
- `validation-benchmarks` - Tracks baselines, timings, correctness checks, and acceptance thresholds for fast particle dynamics.
- `host-integration` - Tracks CLI/build/test integration, compatibility with existing GPU paths, and boundary amendments.

## Key references

See `.ai/implementations/blast-from-the-past/references/index.md`. Headline items:

- Prabhu - human/internal reference; details to be captured in reference notes.
- NVIDIA Warp documentation - Confirm with team: add exact URL/version before using API details as decision evidence.

## Milestones

- Define measurable performance and correctness targets - target date: Confirm with team.
- Establish baseline GPU NNPS and particle-dynamics benchmark - target date: Confirm with team.
- Prototype Warp-backed path inside approved boundary - target date: Confirm with team.

## Stakeholders

- @kunalpuri-prediqt - implementation owner and user handle for closeouts.
- @prabhu - commit reviewer and key reference.
