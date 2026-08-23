---
type: plan
id: 2026-06-18_warp-grid-direct-summation-step-paths
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-18T14:00:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps]
adr: ADR-0004
within_boundary: true
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
---

# Plan: Grid-direct the summation Euler/KDK step paths

## Goal

Remove flat-cache builds from all device step paths. The continuity PEC path is
already grid-direct (ADR-0004); extend grid-direct to the summation Euler
(`wc_sph_euler_step`) and KDK leapfrog (`_compute_wcsph_acceleration`) paths. A
pure flat->grid neighbor-source switch, NO fusing, preserving the exact
composition (pressure-gradient overwrite -> viscosity additive), so the only
numerical change is neighbor visitation order (the fp32 reorder already
validated for continuity). After this, `build_neighbor_cache_gpu` is used only
by the host `get_nearest_particles` query API, `compute_neighbor_sum`, and the
flat-mode oracle/cross-array tests.

## Approach (`warp_sph.py`)

1. Add `neighbor_mode='flat'` to the five standalone equation helpers
   (`compute_summation_density`, `compute_continuity`, `compute_pressure_gradient`,
   `compute_artificial_viscosity`, `compute_xsph_correction`), threaded to
   `_run_equation_group`. Default flat keeps the oracle/cross-array tests
   unchanged.
2. `_compute_wcsph_acceleration` and `wc_sph_euler_step`: pass
   `neighbor_mode='grid'` to summation density, pressure gradient, viscosity
   (accumulate), and the continuity-mode branch. Keep separate launches (no
   fusing).
3. The summation branch of `wc_sph_leapfrog_step`: pass `neighbor_mode='grid'`
   to its `compute_wcsph_adaptive_timestep` and `compute_xsph_correction` calls.

## Tests / validation

- `test_warp_wc_sph_euler_step_matches_cpu_expected_state` and
  `test_warp_wc_sph_leapfrog_step_matches_cpu_expected_state` still pass
  (grid-direct fp32 reorder; loosen a tolerance only if the shift is
  demonstrably fp32-scale, as accepted for continuity).
- Add assertions: the summation Euler and KDK leapfrog steps build zero flat
  caches (monkeypatch `build_neighbor_cache_gpu`).
- Continuity adaptive `nx=100` guard unchanged (`1393` steps); focused suite
  green; `validate-memory`; adversarial-review workflow before sign-off.

## Success criteria

- No device step path builds a flat neighbor cache; `build_neighbor_cache_gpu`
  remains only for the host query API, `compute_neighbor_sum`, and flat-mode
  oracle/cross-array tests. Summation Euler/KDK parity holds at fp32 scale.

## Risks

- fp32 visitation-order shift on the summation parity tests (bounded ~1e-6,
  validated pattern). The summation-density grid cutoff matches the flat cache's
  (`radius_scale*h`), so the neighbor set is identical.

## Out of scope

- Fusing the summation accel; deleting `build_neighbor_cache_gpu`; periodic
  distance; the production results report.

## Approval

- [x] Plan posted in chat and approved
- Approved by: @kunalpuri-prediqt at 2026-06-18T14:00:00 CEST
- Approval, verbatim quote:
  > APPROVED
