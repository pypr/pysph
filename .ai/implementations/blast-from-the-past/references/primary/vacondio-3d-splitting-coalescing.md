---
type: reference-note
id: vacondio-3d-splitting-coalescing
created: 2026-07-06T10:45:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, particle-memory, validation-benchmarks]
---

# Reference: Vacondio et al. 3D splitting and coalescing

## Citation

R. Vacondio, B. D. Rogers, P. K. Stansby, and P. Mignosa, “Variable
resolution for SPH in three dimensions: Towards optimal splitting and
coalescing for dynamic adaptivity,” CMAME 300 (2016) 442--460,
DOI `10.1016/j.cma.2015.11.021`.

## TL;DR

Directly studies conservative 3D weakly-compressible SPH refinement. It
compares 8-vertex cubic, 14-vertex cubic-plus-face-center, 12-vertex
icosahedral, and 20-vertex dodecahedral arrangements, with and without a
daughter at the parent location. The reported optimum is the 12-vertex
icosahedron plus one central daughter (13 total); kernel choice has little
effect on the optimal stencil.

## Key claims

- The split/coalescence formulation conserves mass and momentum while choosing
  parameters to minimize density interpolation error.
- A daughter should remain at the parent location regardless of stencil.
- The icosahedral arrangement is the best of the tested 3D configurations.
- Cubic, quintic, and Wendland kernels show similar density-error ranking.

## Bearing on blast-from-the-past

This is the primary candidate for the P0 3D stencil. A 14-or-15-slot operation
is much more expensive and allocation-heavy than binary or eight-child splits,
so the Warp kill test must compare density error per resulting active particle,
not only minimum error.

## Equations / algorithms / APIs to use

- For the Wendland case, the paper selects shell radius
  `epsilon*h_parent = 0.65*h_parent` and daughter smoothing length
  `alpha*h_parent = 0.70*h_parent`. The 12 shell masses are equal; the central
  mass differs and all 13 mass fractions are obtained from the constrained
  density-error minimization.
- Pairwise conservative coalescence: mass-weighted position/velocity and a
  smoothing length selected to minimize density error.

## Questions raised

- Can iterative merge after splitting recover most of the accuracy with a
  cheaper GPU stencil?
- Does the current Wendland routing reproduce the paper's density-error ranking
  at fp32?

## Verdict

Use as the 3D scientific authority for candidate selection. The open post-print
has now been ingested; do not lock the stencil until the NumPy reproduction
matches the paper's error and mass-ratio table.
