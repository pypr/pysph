---
aspect: particle-memory
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T08:34:00 CET
status: active
---

# Aspect: particle-memory

## What this aspect covers

ParticleArray/device data ownership, transfer semantics, dtype/precision, and compatibility with existing PySPH device helpers.

## Current understanding

ParticleArray currently owns host `BaseArray` storage for properties and constants, while non-cython GPU backends attach `DeviceHelper` as a mirror. The important invariants for a Warp port are:

- Property storage is structure-of-arrays: one flat array per property, with optional fixed per-particle stride.
- `tag`, `pid`, and `gid` are baseline properties; `tag` drives Local/Remote/Ghost behavior.
- `align_particles()` partitions Local particles first and updates `num_real_particles`.
- Constants are fixed-size named arrays and do not resize with particle count.
- Serialization/dummy-particle creation depends on property type/default/stride metadata.

See `.ai/implementations/blast-from-the-past/spec/particle-array/`.

The current Warp elliptical-drop path keeps repeated-step particle state on
device. `wc_sph_leapfrog_step(push=False)` avoids host pushes during the loop;
adaptive dt pulls only the reduced scalar timestep needed as the next launch
parameter, and the runner pulls full arrays only for explicit final metrics,
checkpoint `.npz` output, and side-by-side plots.

The PySPH Application parity path now also keeps density evolution
device-authoritative. `density_mode='continuity'` stores `x0/y0/z0`,
`u0/v0/w0`, and `rho0` on device, computes `arho` on device, and updates `rho`
through WCSPH PEC-style stages without full host pulls/pushes inside the
repeated stepping loop. The only per-step host handoff remains the scalar
adaptive timestep. The PySPH-like `n_damp` and output-time landing policy is
applied to that scalar only; it does not introduce particle-array transfers
inside the step loop.

## Key sub-topics

- ParticleArray property ownership.
- Device helper compatibility.
- Float/double precision choices.
- Strided property gather/scatter behavior.
- Host/device sync authority for `get()`, output, and Cython callers.
- Scalar-only adaptive timestep handoff versus full-array checkpoint pulls.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: particle-memory` and `scope: global`.

## Cross-aspect dependencies

- Influences: `warp-backend`, `gpu-nnps`, and `validation-benchmarks`.
