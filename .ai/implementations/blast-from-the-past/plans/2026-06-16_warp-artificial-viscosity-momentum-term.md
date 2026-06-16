---
type: plan
id: 2026-06-16_warp-artificial-viscosity-momentum-term
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T13:30:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
within_boundary: false
---

# Plan: Warp artificial viscosity momentum term

## Goal

Add Monaghan-style artificial viscosity to the current Warp WCSPH momentum path
and expose it through the elliptical-drop runner so the next GPU runs have the
first missing stabilizing term.

## Context

The committed Warp elliptical-drop runner advances finite prototype states but
still uses only inviscid pressure-gradient acceleration. PySPH's
`pysph.sph.wc.basic.MomentumEquation` combines pressure and artificial
viscosity:

```text
a_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) * grad(W_ij)
```

where `Pi_ij` is nonzero only for approaching pairs:

```text
v_ij dot x_ij < 0
mu_ij = HIJ * (v_ij dot x_ij) / (R2IJ + EPS)
Pi_ij = (-alpha*c_ij*mu_ij + beta*mu_ij^2) * RHOIJ1
c_ij = 0.5*(c_i + c_j)
RHOIJ1 = 1/rho_ij = 2/(rho_i + rho_j)
```

The current Warp prototype does not yet carry `cs`; this slice can use constant
`c0` for both particles as a first WCSPH-equivalent approximation matching the
elliptical-drop configuration's constant reference speed of sound.

## Approach

1. Extend the Warp pressure-gradient kernel path to optionally include
   artificial viscosity with parameters `alpha`, `beta`, `c0`, and `eps`.
2. Keep the existing inviscid default behavior unchanged by defaulting
   `alpha=0.0`, `beta=0.0`.
3. Add a convenience function or parameterized path so
   `wc_sph_leapfrog_step()` can request the viscous momentum term.
4. Add CPU-reference tests for the artificial-viscosity acceleration in a small
   fixture with both approaching and separating particle pairs.
5. Add CLI options to the Warp elliptical-drop runner: `--alpha` and `--beta`,
   then run at least the existing smoke case with `alpha=0.1`, `beta=0.0`.
6. Update experiment docs and memory with the new result.

## Files expected to change

- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_sph.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- current/aspect/daily/session memory updates

Boundary note: these files are part of the approved Python Warp prototype
surface, but the memory validator treats the `warp_*.py` boundary entry
literally rather than as a glob. Mark this plan as `within_boundary: false` and
call it out again in review.

## Tests / validation

- `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- `bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh`
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past`

## Risks

- Constant `c0` is an approximation until a `cs` property/Tait EOS path is
  added.
- The current CubicSpline kernel still differs from PySPH elliptical-drop's
  Gaussian kernel.
- This is still not a full validation against the analytical elliptical-drop
  result.

## Out of scope

- Tait EOS and per-particle sound speed.
- XSPH correction.
- Gaussian kernel support.
- PySPH `Application`/`Solver` integration.
- PR creation.

## Estimated effort

One focused implementation session.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T13:35:58 CEST
- Approval, verbatim quote:
  > APPROVED
