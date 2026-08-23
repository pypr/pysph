---
type: review
date: 2026-06-17
user: @kunalpuri-prediqt
agent: claude
plan: .ai/implementations/blast-from-the-past/plans/2026-06-17_warp-fuse-neighbor-loop-equations.md
adrs: [ADR-0003]
aspects_touched: [warp-backend, gpu-nnps, validation-benchmarks]
host_files: [pysph/base/warp_codegen.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_codegen.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Warp fuse neighbor-loop equations via code generation

## Diff summary

- New module `pysph/base/warp_codegen.py`: a dynamic Warp equation-group code
  generator (ADR-0003). `WarpEquation` blocks declare source/dest/out arrays,
  the shared per-pair quantities they need, and `initialize`/`loop`/`post_loop`
  source snippets; `build_group_kernel` unions the signature, emits the shared
  geometry once, inlines each block's loop into one neighbor traversal
  accumulating into shared `_acc_<out>` registers, materializes the templated
  source (`linecache` + `exec`), wraps it with `wp.Kernel(func=..., source=...)`,
  and caches by `(ordered equation signatures, dtype)`.
- `pysph/base/warp_sph.py`:
  - imports the generator; adds `_WARP_DEVICE_FUNCS` (device `wp.func`s seeded
    into generated kernels);
  - adds four `WarpEquation` blocks (`ContinuityEquation`, `PressureGradient`,
    `ArtificialViscosity`, `XSPHCorrection`) and `_WCSPH_CONTINUITY_BLOCKS`,
    ported from the hand-written kernels with matching math;
  - adds `compute_wcsph_accel_continuity` (single launch + one sync over the
    fused group) and factors EOS into `_apply_wcsph_eos`;
  - rewires `_wc_sph_pec_continuity_step` to EOS + one fused launch per
    half-stage; removes the now-dead `_compute_wcsph_xsph` helper.
- Tests: new `pysph/base/tests/test_warp_codegen.py` (generator mechanism) and
  two additions to `test_warp_sph.py` (fused-vs-separate parity; single fused
  launch per stage with zero per-equation-helper calls).
- ADR-0003 accepted; plan, experiment, aspect, current, daily, session-log
  memory updated; new summary folder `million-cpu-gpu-10step-fused-eqns/`.

## Aspects touched and host files modified

- Aspects: `warp-backend`, `gpu-nnps`, `validation-benchmarks`.
- Host files: `pysph/base/warp_codegen.py` (new), `pysph/base/warp_sph.py`,
  `pysph/base/tests/test_warp_codegen.py` (new),
  `pysph/base/tests/test_warp_sph.py`.

## Behavioral / numerical changes

- The continuity-density PEC half-stage now runs one generated neighbor-loop
  kernel (continuity + pressure gradient + Monaghan viscosity + XSPH) instead of
  four separate launches. Cache builds per step remain 2.
- The summation-density path, the Euler step, the standalone per-equation
  helpers (kept as the trusted oracle), and the adaptive `_wcsph_dt_factors`
  traversal are unchanged.
- Fusion follows PySPH group semantics: each block's `loop` runs per pair inside
  one neighbor loop, accumulating into shared `_acc_<out>` registers. This
  reorders fp32 accumulation versus the two-kernel (pgrad-then-viscosity) path.
- Million-particle fixed-step (`nx=565`, 1,002,885 particles, 10 steps),
  segmented per-step: equation launches/step 8 -> 2; equation-kernel time/step
  ~0.064-0.088 s -> 0.011-0.014 s (~5-6x); step wall (steady) 0.098-0.138 s ->
  0.076-0.098 s (~25%); cache build ~0.034-0.046 s (now the dominant cost).
- 10-step headline wall is overhead/IO-bound (Warp init + 1M mgrid + 282 MB npz
  write); warm samples 3.77-6.04 s (best 3.77 s = `15.25x` vs CPU 57.48 s) vs
  the prior cache-reuse 4.51 s. Per-step compute is the meaningful metric.
- Parity vs prior separate-kernel Warp: positions/density/pressure identical to
  fp32 print precision; kinetic-energy delta `-6.4e-09`. Vs CPU baseline:
  `x ~1e-7`, `rho ~1e-8`, `kinetic_energy 8.5e-06` (same as cache-reuse). Finite.
- Adaptive `nx=100` resolved guard (Warp-only): `1393` steps (identical to
  committed), all finite; shape deltas `~4.8e-07`, density `~1e-06`,
  kinetic-energy `1.45e-04` vs committed Warp. Adaptive path now 2 traversals/
  stage (vs 5); Warp wall `23.63 s` -> `10.83-14.33 s` (cross-session, same step
  count).

## Tests / validation run

```text
$ python -m py_compile pysph/base/warp_codegen.py pysph/base/warp_sph.py pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py
pass
```

```text
$ python -m pytest -q pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
47 passed, 2 warnings in 4.26s
```

```text
$ python .../warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output .../million-cpu-gpu-10step-fused-eqns/warp/million-warp.npz
warm-cache wall samples: 6.04, 5.94, 4.89, 3.77, 4.05 s (best 3.77; cold first-compile 6.15)
segmented per-step: equation launches 2, equation time 0.011-0.014 s, step wall 0.076-0.098 s, cache 0.034-0.046 s
fused vs prior-Warp: positions/density/pressure identical to fp32 print precision; KE delta -6.43e-09
fused vs CPU: x ~1e-7, rho ~1e-8, kinetic_energy 8.52e-06; all_finite True
```

```text
$ adaptive nx=100 resolved guard (Warp-only, pysph timestep policy)
steps 1393 (committed 1393), wall 10.83 / 14.33 s, all_finite True
deltas vs committed Warp: axis_major 4.77e-07, axis_minor 4.77e-07, rho_min -9.54e-07, rho_max 1.07e-06, kinetic_energy 1.45e-04
```

## validate-memory.py

```text
validate-memory: PASS
```

```text
$ git diff --check -- pysph/base/warp_codegen.py pysph/base/warp_sph.py pysph/base/tests/test_warp_codegen.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past
clean
```

## Boundary amendment

- implementation.md boundary section updated: n-a (`pysph/base/warp_*.py` and
  `pysph/base/tests/test_warp_*.py` are already in the boundary; the new
  `warp_codegen.py` / `test_warp_codegen.py` match those globs).
- Amendments log entry: n-a

## Visual aid

Segmented per-step comparison (million-particle fixed step, steady state):

```text
                    cache reuse (4 launches/stage)   fused (1 launch/stage)
equation launches      ########  (8)                  ##  (2)
equation time/step     ~0.064-0.088 s                 0.011-0.014 s   (~5-6x)
step wall/step         0.098-0.138 s                  0.076-0.098 s   (~25%)
cache build/step       0.036-0.052 s                  0.034-0.046 s   (now dominant)
```

## Risks

- fp32 accumulation reordering shifts results ~1e-7 to ~1e-6 (KE ~1e-4 in the
  adaptive run); within parity tolerances and far below the worst case. The
  adaptive guard confirms identical step count (1393).
- Transitional duplication: the four equations exist both as generator blocks
  and as hand-written kernels (kept as oracle / summation path). Migration is
  the ADR-0003 follow-up.
- Generated-source ergonomics: a malformed snippet surfaces as a Warp compile
  error; mitigated by the small documented vocabulary and the
  generated-vs-oracle parity tests.
- The neighbor-cache build is now the dominant per-step cost (~45-50%); next
  optimization target, not addressed here.

## Unresolved questions

- Should the standalone per-equation helpers and the summation/adaptive paths be
  migrated onto the generator next (retiring the duplicated hand kernels), or
  should the cache-build cost be attacked first now that it dominates?
- The 10-step headline wall is overhead/IO-bound and noisy; do we want a
  longer-step headline (or the adaptive run) as the canonical performance metric?
- Raw million-particle HDF5/NPZ dumps are large; only summary JSON is retained.
  Unrelated untracked `CODEBASE_UNDERSTANDING.md` remains untouched.

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > LGTM
