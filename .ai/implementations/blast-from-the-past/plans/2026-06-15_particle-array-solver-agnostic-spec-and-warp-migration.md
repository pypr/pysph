---
type: plan
id: 2026-06-15_particle-array-solver-agnostic-spec-and-warp-migration
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-15T00:00:00 CET
status: approved
aspects: [particle-memory, cython-boundary, warp-backend, host-integration, validation-benchmarks]
host_files: [pysph/base/particle_array.pxd, pysph/base/particle_array.pyx, pysph/base/utils.py, pysph/base/device_helper.py, pysph/base/tests/test_particle_array.py, pysph/base/tests/test_utils.py]
within_boundary: false
---

# Plan: particle array solver agnostic spec and warp migration

## Goal

Produce a solver-agnostic implementation specification for PySPH's base particle-array data structures, then use that spec to decide and plan the first NVIDIA Warp migration step for particle arrays.

## Context

User request: start the first implementation using NVIDIA Warp for particle array classes, using the attached solver-agnostic spec prompt to review base array data structures first. The top-level `CODEBASE_UNDERSTANDING.md` is a whole-codebase reference document. The local environment on `prediqt-02` can be activated with `source $HOME/prediqt/activate`.

The attached prompt requires every spec claim to be labeled `[OBSERVED]`, `[INFERRED]`, or `[UNKNOWN]`, with `file:line` references for observed claims. It also requires starting with `open-questions.md`, tracing entry points/call graph, reading tests/docs before deep implementation files, and writing a `spec/` directory of Markdown files.

Boundary note: this plan reads tests, docs, and Python helpers outside the `.pxd/.pyx` implementation boundary as discovery evidence only. It does not modify host application code.

## Approach

Phase 1 - solver-agnostic reference spec, no host code migration:

1. Read relevant tests/docs first:
   - `pysph/base/tests/test_particle_array.py`
   - `pysph/base/tests/test_utils.py`
   - relevant docs/tutorial references to `ParticleArray`
   - top-level `CODEBASE_UNDERSTANDING.md`
2. Trace one representative lifecycle end-to-end:
   - particle array creation through `get_particle_array`
   - property/constant allocation
   - host/device helper attachment
   - output serialization/readback path where relevant
3. Read core implementation:
   - `pysph/base/particle_array.pxd`
   - `pysph/base/particle_array.pyx`
   - `pysph/base/utils.py`
   - `pysph/base/device_helper.py`
   - targeted references from NNPS/solver/output only where needed for call placement and host contract.
4. Create spec files under `.ai/implementations/blast-from-the-past/spec/particle-array/`:
   - `open-questions.md`
   - `00-overview.md`
   - `01-mesh-geometry.md`
   - `02-timeline.md`
   - `03-data-structures.md`
   - `04-boundary.md`
   - `05-parallelism.md`
   - `06-host-contract.md`
   - `07-variants.md`
   - `08-interfaces.md`
   - `09-verification.md`
   - `10-porting.md`
   - `glossary.md`
5. Update aspect context/open questions for durable discoveries.
6. If the spec implies an architectural choice for Warp memory ownership, create an ADR before any migration code.

Phase 2 - Warp migration planning, gated:

1. Activate the environment with `source $HOME/prediqt/activate` for runtime checks.
2. Probe availability/version of NVIDIA Warp and existing PySPH build/test state.
3. Create a separate implementation plan for the first code migration step, scoped to particle-array/device memory only.
4. Do not modify host application code in this plan unless the user explicitly approves a follow-up migration plan.

## Files expected to change

- `.ai/implementations/blast-from-the-past/spec/particle-array/**`
- `.ai/implementations/blast-from-the-past/aspects/particle-memory/*`
- `.ai/implementations/blast-from-the-past/aspects/cython-boundary/*`
- `.ai/implementations/blast-from-the-past/aspects/warp-backend/*`
- Possibly `.ai/implementations/blast-from-the-past/decisions/*` if an ADR is warranted.
- No PySPH host code in Phase 1.

## Tests / validation

- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai AGENTS.md`
- For environment probing only after spec: `source $HOME/prediqt/activate && python -c "import warp; print(warp.__version__)"` if Warp is expected to be installed.
- Existing host tests are not required for Phase 1 because no host code changes are made; later migration plans must include focused tests.

## Risks

- The attached prompt is solver-agnostic CFD-oriented, while `ParticleArray` is infrastructure rather than a numerical flux/closure module; sections about governing equations, mesh, and boundary conditions will contain many `[UNKNOWN]` or "not applicable to this data structure" entries.
- `CODEBASE_UNDERSTANDING.md` is currently untracked; it can be read as user-provided context but should not be committed unless explicitly requested.
- Runtime Warp availability may differ from expected environment state.
- A premature migration could bake in the wrong ownership model; the spec/ADR gate is intended to avoid that.

## Out of scope

- Rewriting `ParticleArray` or `DeviceHelper` in this plan.
- Changing build dependencies or package metadata in this plan.
- Migrating NNPS, equation evaluation, integrators, or solver loop in this plan.
- Committing `CODEBASE_UNDERSTANDING.md`.

## Estimated effort

M/L - the spec is multi-file and citation-heavy; migration code requires a follow-up plan.

## Approval

- [ ] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-15T07:49:02 CEST
- Approval, verbatim quote:
  > APPROVED
