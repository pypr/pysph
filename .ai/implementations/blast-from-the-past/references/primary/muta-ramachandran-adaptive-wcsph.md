---
type: reference-note
id: muta-ramachandran-adaptive-wcsph
created: 2026-07-06T10:45:00 CEST
author: @kunalpuri-prediqt
kind: primary
status: assessed
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks]
---

# Reference: Muta--Ramachandran efficient adaptive WCSPH

## Citation

Abhinav Muta and Prabhu Ramachandran, “Efficient and Accurate Adaptive
Resolution for Weakly-Compressible SPH,” CMAME 2022,
DOI `10.1016/j.cma.2022.115019`; manuscript `arXiv:2107.01276`; source
`https://gitlab.com/pypr/adaptive_sph` (audited at repository HEAD on
2026-07-06).

## TL;DR

Provides the closest PySPH-native adaptive workflow: background particles set
spatial target mass, mutually closest particles merge in parallel, particles
split and are then repeatedly merged, shifting regularizes the distribution,
and smoothing length is reset from local average mass to control neighbor
count. It is not a ready 3D dam-break implementation.

## Key claims

- Geometry- and solution-driven target resolution can be represented by
  background particles.
- Split/merge can be parallel; adaptation typically runs every 1--10 steps.
- Optimizing `h` from neighboring mass avoids the excessive support and
  neighbor counts of older APR methods.
- The paper reports length-scale variation up to 1:250 and substantially fewer
  particles than uniform-resolution comparisons.

## Bearing on blast-from-the-past

Use the workflow and validation ideas, not the implementation verbatim. The
paper validates 2D EDAC-style flows without a free surface. The source confirms
that automatic background adaptation asserts `dim == 2`, several mass/spacing
updates carry `FIXME in 3D`, and its GPU lifecycle path is explicitly not
implemented. Its `AdaptiveResolution` does contain a binary 3D split helper,
but that is not the paper's fully validated automatic method.

## Equations / algorithms / APIs to use

- Target thresholds: `m_max = 1.05*m_ref`, `m_min = 0.5*m_ref` as a candidate,
  subject to 3D kill tests.
- Parallel merge ownership: mutually closest eligible pair; retain lower stable
  ID; mass-weighted position, velocity, and scalar properties.
- Smoothing length from local mass scale: `h = C*(m/rho)^(1/d)` after
  adaptation.
- Adapt -> iterative merge -> shift -> first-order property correction -> NNPS
  rebuild ordering.

## Questions raised

- Which 3D split stencil minimizes density error with the current Wendland
  kernel and practical neighbor count?
- Which variable-`h` correction is required for the continuity-density WCSPH
  formulation?
- How must shifting be limited at a violent free surface and fixed obstacle?

## Verdict

Adopt as the PySPH process/reference baseline. Do not claim it validates the
planned 3D WCSPH implementation.
