---
type: review
date: 2026-06-18
user: @kunalpuri-prediqt
agent: claude
plan: .ai/implementations/blast-from-the-past/plans/2026-06-18_warp-grid-direct-summation-step-paths.md
adrs: [ADR-0004]
aspects_touched: [warp-backend, gpu-nnps]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Grid-direct the summation Euler/KDK step paths

## Diff summary

- `pysph/base/warp_sph.py`:
  - The five standalone equation helpers (`compute_summation_density`,
    `compute_continuity`, `compute_pressure_gradient`,
    `compute_artificial_viscosity`, `compute_xsph_correction`) gain
    `neighbor_mode='flat'` (default), threaded to `_run_equation_group`. Flat
    default keeps the oracle/cross-array tests and `compute_neighbor_sum`
    unchanged.
  - `_compute_wcsph_acceleration` (summation KDK) and `wc_sph_euler_step` now
    pass `neighbor_mode='grid'` to summation density, pressure gradient,
    viscosity (accumulate), and the continuity-mode branch.
  - The summation branch of `wc_sph_leapfrog_step` passes `neighbor_mode='grid'`
    to its `compute_wcsph_adaptive_timestep` and `compute_xsph_correction` calls.
  - No fusing: the pressure-gradient(overwrite) -> viscosity(add) composition is
    preserved; the only numerical change is neighbor visitation order.
- Tests: `test_warp_summation_step_paths_build_no_flat_neighbor_cache` asserts
  the summation Euler and KDK leapfrog steps build zero flat caches (and consult
  the grid).

## Behavioral / numerical changes

- All device step paths are now grid-direct (continuity was already; this adds
  the summation Euler + KDK leapfrog paths). `build_neighbor_cache_gpu` is now
  used only by the host `get_nearest_particles` query API, `compute_neighbor_sum`,
  and the flat-mode oracle/cross-array tests.
- The summation-density grid cutoff (`radius_scale*h`) matches the flat build's
  neighbor set, so rho and accelerations are unchanged apart from fp32
  visitation-order reorder (~1e-7). The continuity PEC path is untouched.

## Tests / validation run

```text
$ python -m pytest -q pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
53 passed, 2 warnings

(summation Euler + KDK leapfrog CPU-parity tests pass under grid-direct with no
tolerance changes; new no-flat-cache assertion passes.)
```

```text
$ continuity adaptive nx=100 guard (unchanged path, sanity)
steps 1393 (identical), KE 7797.7071, finite
```

## validate-memory.py

```text
validate-memory: PASS
```

## Adversarial pre-review

A 4-dimension workflow (accel composition, neighbor membership, grid lifecycle,
flat-default preservation), each finding verified by an independent skeptic.
Result: **0 confirmed, 0 raised** -- clean diff.

## Risks

- fp32 visitation-order shift on the summation parity tests; bounded (~1e-7) and
  within the existing 1e-5 tolerances (no loosening required).
- Grid mode requires a grid-capable NNPS (`UniformGridWarpNNPS`); the WCSPH step
  paths only ever use that NNPS.

## Unresolved questions

- `build_neighbor_cache_gpu` could now be narrowed further or moved behind the
  host-query API module; deferred (it still legitimately backs the oracle tests,
  cross-array queries, and `compute_neighbor_sum`).

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > LGTM
