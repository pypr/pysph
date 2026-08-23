---
type: decision
id: ADR-0002
date: 2026-06-15
author: @kunalpuri-prediqt
scope: particle-memory
status: Accepted
supersedes: []
relates_to: [ADR-0001]
depends_on: []
conflicts_with: []
---

# ADR-0002: Warp Device Mirror For ParticleArray

## Context

The ParticleArray spec shows that `ParticleArray` currently owns host `BaseArray` storage, exposes Cython declarations, and delegates non-cython device behavior to `DeviceHelper`.

The public contract includes `get_carray()`, NumPy/readback paths, pickle/output metadata, strided properties, constants, and Local-first tag alignment.

Warp `1.14.0` imports in the active environment.

## Decision

Implement NVIDIA Warp first as a DeviceHelper-like mirror for ParticleArray rather than replacing host `BaseArray` ownership.

## Rationale

This keeps the existing Cython and host API stable while allowing Warp kernels to prove parity for device creation, push/pull, alignment, and particle mutation primitives.

## Alternatives considered

- Replace ParticleArray storage with Warp arrays. This may be faster eventually, but it risks breaking Cython callers and output/restart compatibility before the backend contract is proven.
- Hide Warp under the existing CUDA backend. This may reduce API surface, but it makes backend selection and comparison against current CUDA/Compyle behavior less explicit.

## Consequences

- A new helper or helper mode must preserve current push/pull and metadata semantics.
- Performance work begins with mirror overhead included.
- A later ADR can revisit authoritative Warp ownership after behavior parity and benchmarks exist.

## Follow-ups

- Implement the focused Warp mirror prototype plan.
