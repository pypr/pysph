---
type: reference-note
id: yang-gpu-adaptive-particle-refinement
created: 2026-07-06T10:45:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks]
---

# Reference: Yang et al. GPU adaptive particle refinement

## Citation

Q. Yang, F. Xu, Y. Yang, Z. Dai, and J. Wang, “A GPU-accelerated adaptive
particle refinement for multi-phase flow and fluid-structure coupling SPH,”
Ocean Engineering 279 (2023) 114514,
DOI `10.1016/j.oceaneng.2023.114514`.

## TL;DR

Demonstrates that dynamic APR can remain GPU-resident using explicit resource
management, refinement kernels, and particle shifting. It is architectural
evidence, not a formulation oracle for the current single-phase WCSPH driver.

## Key claims

- GPU-side dynamic resource management addresses changing particle count.
- Axis-aligned refinement plus a virtual-fine-particle shifting treatment can
  remain accurate and stable in multi-phase/FSI examples.
- Adaptive cases reduce computational cost compared with uniform particles at
  comparable precision.

## Bearing on blast-from-the-past

Supports the planned capacity-managed device pool, scan allocation, and
batched adaptation checkpoint. The paper uses a different Riemann-based SPH
model and does not justify copying its physics into the current backend.

## Equations / algorithms / APIs to use

- Separate capacity from active count.
- Reuse slots released by coalescence before growing storage.
- Regularize transition particles after refinement.

## Questions raised

- Which pool growth and compaction policy maps best to Warp arrays?
- Can stable IDs and deterministic merge ownership be kept without serial
  allocation?

## Verdict

Adopt the resource-management principles; independently validate all WCSPH
physics and split parameters.
