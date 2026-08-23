---
type: review
date: 2026-06-18
user: @kunalpuri-prediqt
agent: claude
plan: .ai/implementations/blast-from-the-past/plans/2026-06-18_warp-migrate-neighbor-kernels-onto-generator.md
adrs: [ADR-0003]
aspects_touched: [warp-backend, gpu-nnps]
host_files: [pysph/base/warp_codegen.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_codegen.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Migrate remaining neighbor-loop kernels onto the generator (ADR-0003 follow-up)

## Diff summary

- `pysph/base/warp_codegen.py`:
  - `accumulate_outputs` option on `generate_group_source`/`build_group_kernel`
    (part of `_cache_key`): seeds each `_acc_<out>` from the existing
    `d_<out>[i]` instead of zero, so a group adds to (read-modify-writes) the
    destination arrays. Needed for the standalone artificial-viscosity term,
    which composes onto a prior pressure-gradient acceleration.
  - Deterministic kernel `func_name`: derived from an md5 of the structural
    cache key instead of `len(_KERNEL_CACHE)`. The old call-order-dependent name
    changed the generated source whenever the build order shifted, defeating
    Warp's on-disk kernel cache and forcing a full cold recompile every session.
- `pysph/base/warp_sph.py`:
  - New blocks `SummationDensity` (overwrite `rho`) and `WcsphCflFactor`
    (`dt_cfl` = neighbor `wp.max` reduction of `|hij*(vij.xij)/rij^2| + c0`;
    `dt_force` = `au^2+av^2+aw^2` in `post_loop`).
  - `_run_equation_group(...)`: shared launcher binding inputs in the
    generator's canonical order, flat or grid, with optional
    `accumulate_outputs`; self- and cross-array (`src != dst`).
  - Repointed `compute_summation_density`, `compute_continuity`,
    `compute_pressure_gradient`, `compute_artificial_viscosity` (accumulate),
    `compute_xsph_correction`, and `compute_wcsph_adaptive_timestep` onto the
    generator; refactored `compute_wcsph_accel_continuity` onto the shared
    launcher.
  - Retired the ~14 duplicated hand `@wp.kernel`s (`_summation_density`,
    `_continuity`, `_pressure_gradient`, `_artificial_viscosity`,
    `_xsph_correction`, `_wcsph_dt_factors{,_grid}`, f32+f64). EOS, integrator,
    and dt init/reduce/finalize reduction kernels are unchanged.
  - Fail-fast guards: the repointed helpers reject a non-canonical
    `out_prop`/`out_props` (the generated path writes the block's canonical
    array names) instead of silently writing the wrong array.
- Tests: `test_accumulate_outputs_adds_to_existing_output` (codegen);
  `test_warp_equation_helpers_reject_custom_output_names` (guards);
  `test_warp_fused_accel_matches_separate_helpers` re-commented as a
  fusion-consistency check (both sides now generator-backed).

## Behavioral / numerical changes

- Consolidation: the generator is now the single source for every neighbor-loop
  kernel. The summation path's composition is preserved exactly (pressure
  gradient overwrites `au`, viscosity adds via `accumulate_outputs=True`, then
  continuity). The continuity hot path uses the same fused grid kernel as
  before (byte-identical source) -- perf-neutral.
- Numerically the single-block generated groups reproduce the retired hand
  kernels; deltas stay at fp32 scale, guarded by the existing per-equation
  CPU-reference and cross-array tests.
- Deterministic naming: the generated source is now stable across processes
  (verified: identical `src_md5` in two independent interpreters). Each kernel
  cold-compiles once per machine, then loads from Warp's disk cache (~20 ms).

## Tests / validation run

```text
$ python -m py_compile pysph/base/warp_codegen.py pysph/base/warp_sph.py <tests>
pass

$ python -m pytest -q pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
52 passed, 2 warnings
```

```text
$ adaptive nx=100 resolved guard (Warp-only, pysph timestep policy)
steps 1393 (committed 1393), all_finite True
kinetic_energy 7797.7070 (relative delta ~8e-9 vs committed), shape/density deltas at fp32 scale

$ million fixed-step segmented profile (nx=565, continuity path)
flat_cache_builds 0; steady step-wall floor 0.059 s (== prior grid-direct); KE identical (7854.1276); finite
```

```text
$ disk-cache fix verification
cross-process generated-source md5 identical (cfl-grid, continuity-grid)
warm guard rerun: both generated modules load (cached) ~20 ms; no recompile (was a one-time 102-184 s cold compile)
```

## validate-memory.py

```text
validate-memory: PASS
```

## Adversarial pre-review

A 6-dimension workflow (accumulate semantics, I/O contract, CFL-factor block,
launch binding, deterministic naming, completeness), each finding verified by an
independent skeptic. Result: 1 confirmed (minor), 1 dismissed (informational:
the harmless `dt_force` double-write).

- Confirmed minor: the repointed helpers kept `out_prop`/`out_props` params used
  for ensure/push/return, but the generated kernel writes only the block's
  canonical names, so a non-default name would silently write zeros (currently
  unreachable -- no caller uses non-defaults). **Addressed**: fail-fast guards
  + a test. Signatures preserved per the plan.
- The deterministic-naming latent bug (disk-cache thrash) was surfaced by the
  perf dimension while investigating a 184 s cold compile and fixed in this
  slice.

## Risks

- Additive vs overwrite: viscosity must use `accumulate_outputs=True`; the
  pressure-gradient-then-viscosity composition is guarded by
  `test_warp_artificial_viscosity_matches_cpu_and_adds_to_acceleration` and the
  fused-vs-separate consistency test.
- fp32 single-block-group vs hand-kernel ordering ~1e-7; within parity
  tolerances (adaptive guard keeps 1393 steps).
- The large fused grid kernel has a slow one-time cold compile (~100-180 s);
  now amortized by the deterministic-name disk cache (one-time per machine).

## Unresolved questions

- Should the summation Euler/KDK path be fused + grid-directed too (the separate
  "grid-direct everywhere" follow-up), now that all its equations are generator
  blocks?
- Should `build_neighbor_cache_gpu` be narrowed to just the host
  `get_nearest_particles` query API now that the device paths are generator-backed?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > LGTM
