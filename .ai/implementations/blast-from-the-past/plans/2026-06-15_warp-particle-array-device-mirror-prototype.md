---
type: plan
id: 2026-06-15_warp-particle-array-device-mirror-prototype
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-15T08:42:00 CET
status: approved
aspects: [particle-memory, warp-backend, cython-boundary, host-integration, validation-benchmarks]
host_files: [pysph/base/particle_array.pyx, pysph/base/particle_array.pxd, pysph/base/device_helper.py, pysph/base/tests/test_particle_array.py, pysph/base/tests/test_device_helper.py]
within_boundary: false
---

# Plan: Warp ParticleArray Device Mirror Prototype

## Goal

Prototype the first NVIDIA Warp-backed ParticleArray path by adding a DeviceHelper-like mirror while preserving existing host `BaseArray` ownership and public ParticleArray behavior.

## Context

The solver-agnostic spec in `.ai/implementations/blast-from-the-past/spec/particle-array/` identifies ParticleArray as a mutable host particle table with an optional device mirror. Warp `1.14.0` imports in the active environment.

ADR-0002 proposes starting with a Warp mirror rather than replacing host storage.

This plan is outside the original `.pxd/.pyx` boundary because a practical mirror prototype likely touches Python helper/test files. No code should begin until this boundary expansion is approved.

## Approach

1. Confirm the desired backend name: tentatively `backend='warp'`.
2. Add a minimal Warp helper behind the existing ParticleArray device-helper boundary.
3. Implement device array creation plus selective/full `push()` and `pull()` for scalar-stride properties and constants.
4. Add Local-first `align_particles()` for `tag` and scalar properties.
5. Extend alignment to strided properties.
6. Add focused tests mirroring existing DeviceHelper push/pull and alignment cases.
7. Initial implementation proved mirror and alignment behavior; user then approved continuing.
8. Add Warp-backed remove, remove-tagged, add, append, extend, and extract behavior through the ParticleArray public methods.
9. Expand tests across the full prototype surface and add tutorial-style documentation.

## Files expected to change

- `pysph/base/particle_array.pyx`
- `pysph/base/particle_array.pxd` if the backend hook needs declaration changes
- `pysph/base/device_helper.py` or a new `pysph/base/warp_device_helper.py`
- `pysph/base/tests/test_device_helper.py` or a new Warp-focused test module
- `.ai/implementations/blast-from-the-past/**`
- `docs/source/tutorial/warp_particle_array.rst`
- `docs/source/index.rst`

## Tests / validation

- `source $HOME/prediqt/activate` only if no environment is already active.
- `python -c "import warp; print(warp.__version__)"`
- Focused Warp helper tests for creation, push, pull, scalar alignment, and strided alignment.
- Expanded Warp helper tests for dtype policy, full sync, readback modes, remove/remove-tagged, add/default fill, append missing properties/constants, empty clone, extract, resize/extend, max, and errors.
- Existing CPU ParticleArray tests to confirm host behavior is unchanged.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai AGENTS.md pysph/base`

## Risks

- Warp dynamic resizing may not match Compyle `Array` semantics directly.
- Implicit host readback behavior can hide stale device data unless synchronization points are explicit.
- Strided alignment is easy to get semantically wrong.
- Adding `backend='warp'` may require changes in backend resolution outside ParticleArray.

## Out of scope

- Migrating NNPS.
- Migrating SPH equations or integrators.
- Replacing host `BaseArray` ownership.
- Replacing host `BaseArray` ownership.
- Making performance claims beyond functional smoke checks.

## Estimated effort

M - small enough for a first prototype, but touches backend selection and test plumbing.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-15T08:03:01 CEST
- Approval, verbatim quote:
  > alright. commit and continue

Commit note: the commit part of that message was superseded by "actually hold off on the commit"; the implementation continuation remains approved.

Scope continuation:

- Approved by: @kunalpuri-prediqt at 2026-06-15T11:57:32 CEST
- Approval, verbatim quote:
  > continue

Documentation/test continuation:

- Approved by: @kunalpuri-prediqt at 2026-06-15T12:12:52 CEST
- Approval, verbatim quote:
  > can you write the tests covering all aspects and also a tutorial style document explaning how to use the new warp particle array class
