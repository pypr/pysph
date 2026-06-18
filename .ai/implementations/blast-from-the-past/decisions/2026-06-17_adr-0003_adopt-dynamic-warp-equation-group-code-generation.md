---
type: decision
id: ADR-0003
date: 2026-06-17
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0002]
depends_on: []
conflicts_with: []
---

# ADR-0003: Adopt dynamic Warp equation-group code generation

## Context

PySPH's defining strength is composability: physics is written as small
`Equation` building blocks (`initialize` / `loop` / `post_loop`), bundled into
`Group`s, and the framework's transpiler generates one fused kernel per group
for each backend (Cython, OpenCL/CUDA via compyle). Fusion is a property of
grouping, not something authored by hand.

The current Warp prototype instead hand-writes one `@wp.kernel` per equation in
both `f32` and `f64`. The cache-reuse slice exposed the cost: a single
continuity-density PEC half-stage launches four separate neighbor-loop kernels
(pressure gradient, artificial viscosity, continuity, XSPH) over the same
~45M-entry neighbor cache, each re-reading neighbor indices and recomputing the
same per-pair geometry. Hand-fusing them would help, but it is a one-off that
must be re-authored for every new equation set and doubles again for `f64`.

A spike on the active machine (RTX 4060, Warp `1.14.0`) confirmed Warp supports
runtime kernel generation: a kernel body assembled at runtime from snippet
lists, materialized as a function (via `linecache` + `exec`) and wrapped with
`wp.Kernel(func=..., source=...)`, JIT-compiled in ~1.5 s (then hash-cached) and
produced correct device results. `wp.Kernel.__init__` exposes an explicit
`source=` parameter, which avoids fragile `inspect.getsource` behavior for
generated functions.

## Decision

Adopt dynamic, composable code generation for Warp SPH kernels as the backend's
kernel-construction model, mirroring PySPH's equation/group transpilation.

A `WarpEquation` block declares the source/dest arrays it reads and the arrays
it writes, and contributes `initialize` / `loop` / `post_loop` source snippets.
A `WarpGroup` of blocks unions the array signature, computes shared per-pair
geometry once (`dx,dy,dz,rij,hij`, kernel gradient/value, velocity diffs),
inlines each block's `loop` snippet into a single neighbor traversal, and emits
one cached, JIT-compiled kernel per `(equation-set structure, dtype, dim)`
signature. The kernel id (cubic/gaussian) stays a runtime argument dispatched by
the existing device `wp.func`s.

The first consumer is the continuity-density PEC half-stage: continuity +
pressure gradient + artificial viscosity + XSPH composed into one generated
kernel.

## Rationale

- Restores the PySPH composition model on the GPU: new equations and new
  combinations fuse automatically instead of requiring a hand-written kernel.
- Fusion (one traversal instead of four) becomes a property of grouping, giving
  the performance win as a side effect of the right abstraction.
- Collapses the `f32`/`f64` duplication into a dtype parameter of the generator.
- The mechanism is proven feasible on the target hardware before committing.

## Alternatives considered

- Hand-write a monolithic fused kernel per solver. Fast to a benchmark number
  and lowest immediate risk, but a one-off that betrays the composition model,
  must be re-authored per equation set, and keeps the `f32`/`f64` split. Likely
  throwaway once the generator lands.
- Compose only via `wp.func` building blocks called from a static outer kernel.
  Reuses device code but still requires a hand-written outer kernel that
  enumerates the equations and recomputes shared geometry per call site; it does
  not give automatic, group-driven fusion.
- Defer GPU fusion and rely on PySPH's existing OpenCL/CUDA transpiler. Out of
  scope: the implementation goal is a Warp-backed path.

## Consequences

- A new code-generation module (e.g. `pysph/base/warp_codegen.py` or an
  addition to `warp_sph.py`) becomes part of the boundary and must be tested:
  generated kernels validated against the existing hand-written per-equation
  kernels (trusted oracle) and CPU references.
- A first cut targets forward-only, single-group, self-interaction
  (`src == dst`) physics with the shared quantities the four equations need.
  Gradient/adjoint support, multi-group/iterated-group orchestration, and
  cross-array source/dest fusion are explicitly deferred.
- Two representations of the four equations coexist transitionally: the new
  blocks (used by the generator) and the existing hand-written kernels (kept as
  oracle and for the summation-density path). A later slice migrates the
  remaining helpers onto the generator and removes the duplication.
- Generated-source ergonomics (readable errors when a snippet is malformed,
  stable cache keys) must be handled or debugging regresses.

## Follow-ups

- Implement plan `2026-06-17_warp-fuse-neighbor-loop-equations` (revised to the
  generator design) with the continuity stage as the first consumer.
- Follow-up slice: express the standalone per-equation helpers and the
  summation-density and adaptive-timestep-factor paths via the generator and
  retire the duplicated hand-written kernels.
