---
type: plan
id: 2026-06-17_warp-fuse-neighbor-loop-equations
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-17T17:40:00 CEST
revised: 2026-06-17T18:05:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, validation-benchmarks]
adr: ADR-0003
host_files:
  - pysph/base/warp_codegen.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp Fuse Neighbor-Loop Equations Via Dynamic Code Generation

## Goal

Build a minimal, composable Warp equation-group **code generator** (ADR-0003)
and make the continuity-density PEC half-stage its first consumer: continuity +
pressure gradient + artificial viscosity + XSPH composed into a single,
dynamically generated, JIT-compiled kernel that traverses the shared neighbor
cache exactly once. Then redo the million-particle headline benchmark.

This realizes PySPH's composition model on the GPU (fusion as a property of
grouping) instead of hand-writing a one-off fused kernel, and it removes the
`f32`/`f64` duplication for the fused path.

Current state after the cache-reuse slice (`nx=565`, 1,002,885 particles, fixed
10 steps): PySPH CPU 57.48 s; Warp 4.51 s; speedup 12.7450110864745x; segmented
step wall 0.098335–0.138290 s with cache time 0.036122–0.052126 s, so the four
separate equation traversals are roughly half of each step's wall time.

## Context

Feasibility is already proven (recorded in ADR-0003): a runtime-assembled kernel
body, materialized via `linecache` + `exec` and wrapped with
`wp.Kernel(func=..., source=...)`, JIT-compiled in ~1.5 s and ran correctly on
the RTX 4060.

In `_wc_sph_pec_continuity_step`, each PEC half-stage currently runs four
separate neighbor-loop launches against one shared `stage_cache`
(`compute_pressure_gradient`, `compute_artificial_viscosity`,
`compute_continuity`, `compute_xsph_correction`), each a launch + device sync,
each re-reading neighbor indices and recomputing the same per-pair geometry. The
Tait/Isothermal EOS feeding pressure/viscosity is a cheap per-particle kernel
that must run before the fused kernel (it reads `p`, `cs`).

## Approach

### 1. Code generator (`pysph/base/warp_codegen.py`)

- `WarpEquation` base class. Each block declares:
  - `src_arrays`, `dst_arrays`, `out_arrays` (property names);
  - which shared per-pair quantities it needs from
    `{dx, dy, dz, rij, hij, grad, wij, vijx, vijy, vijz}`;
  - `scalars` it consumes (e.g. `alpha`, `beta`, `eps`);
  - source snippets `initialize()`, `loop()`, `post_loop()` as strings using a
    small, documented variable vocabulary (`i`, `j`, the shared quantities, the
    `s_<name>`/`d_<name>` arrays, and a `TYPE` token replaced with the dtype).
- `build_group_kernel(equations, dtype, dim_static=None)`:
  - unions arrays into a stable kernel signature
    (`s_*`, `d_*`, `starts`, `lengths`, `neighbors`, `dim`, `kernel_id`,
    scalars, then `d_*` outputs);
  - emits `i = wp.tid()`, each block's `initialize`, one neighbor loop that
    computes only the requested shared quantities **once**
    (`grad = _kernel_dwdq_<dtype>(rij, hij, dim, kernel_id)/(hij*rij)` guarded by
    `rij > tiny`; `wij = _kernel_value_<dtype>(...)`), then each block's `loop`,
    then each block's `post_loop`;
  - `exec`s the templated source into a namespace seeded with `wp` and the
    existing device `wp.func`s (`_kernel_dwdq_f32/f64`, `_kernel_value_f32/f64`),
    registers it in `linecache`, and wraps it with `wp.Kernel(func=..., source=)`;
  - caches the compiled kernel by structural signature key
    `(equation kinds + structural flags, dtype, dim handling)` so each unique
    group compiles once.
- Validate up front that a generated kernel can call the existing module-level
  `wp.func`s (the spike covered only built-ins).

### 2. Equation blocks for the four continuity-stage equations

Port the per-pair math of `_continuity`, `_pressure_gradient`,
`_artificial_viscosity` (Monaghan, pair-averaged `cs`, `vdotx < 0` guard), and
`_xsph_correction` into blocks. Accumulation order is preserved relative to the
hand kernels: pressure gradient and viscosity use distinct accumulators summed
in `post_loop` (`d_au[i] = au_p + au_v`), matching the existing two-kernel
semantics; continuity and XSPH keep their own accumulators.

### 3. Helper + integration

- `compute_wcsph_accel_continuity(nnps, pa_index, alpha, beta, eps, kernel,
  cache, push)`: ensures arrays/`cs`, builds the group kernel for the present
  dtype, does a single `wp.launch` + one `wp.synchronize_device`.
- Factor EOS out into `_apply_wcsph_eos(...)` shared by both paths.
- Refactor `_wc_sph_pec_continuity_step` **only**: per half-stage build the
  cache (unchanged), apply EOS, then call the fused helper instead of the four
  separate helpers; XSPH now comes from the fused kernel so its separate call is
  removed and `use_xsph` is derived from `xsph_eps`.

### 4. Leave untouched

`_compute_wcsph_acceleration` (summation path + Euler step), every existing
per-equation helper and its f32/f64 kernels (kept as the trusted oracle and for
the summation path), and the adaptive `compute_wcsph_adaptive_timestep`
(`_wcsph_dt_factors` still runs its own traversal, reading the fused
`au,av,aw`). The adaptive-factor and summation paths migrate to the generator in
a later slice (ADR-0003 follow-up).

## Files expected to change

- `pysph/base/warp_codegen.py` (new)
- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_codegen.py` (new)
- `pysph/base/tests/test_warp_sph.py`
- `.ai/.../experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- a new compact summary folder
  `.ai/.../experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-fused-eqns/`
- `.ai/.../aspects/warp-backend/context.md`,
  `.ai/.../aspects/validation-benchmarks/context.md`
- `.ai/.../current.md`, `.ai/.../updates/daily/2026-06-17.md`
- session log + review artifact before commit
- ADR-0003 flipped to `Accepted` on approval

Boundary note: `pysph/base/warp_*.py` is already in the implementation
boundary, so `warp_codegen.py` and `test_warp_codegen.py` are within boundary;
the review will record the addition.

## Tests / validation

- New `test_warp_codegen.py`:
  - a generated trivial kernel compiles, runs, and caches (second build is a
    cache hit, not a recompile);
  - a generated kernel can call the existing device `wp.func`s;
  - a generated single-equation kernel matches the corresponding hand-written
    helper on small input.
- `test_warp_sph.py` additions:
  - the generated 4-equation group matches the four separate helpers on the same
    input (tight `allclose` on `arho, au, av, aw, ax, ay, az`);
  - the continuity-density `wc_sph_leapfrog_step()` issues one fused equation
    launch per half-stage and does **not** call the four per-equation helpers
    (monkeypatch-count);
  - the existing two-cache-builds-per-step test still holds;
  - the existing `test_warp_wc_sph_leapfrog_continuity_mode_matches_cpu_pec_state`
    CPU-parity test still passes (now exercising the generated kernel).
- `python -m pytest -q pysph/base/tests/test_warp_sph.py
  pysph/base/tests/test_warp_nnps.py pysph/base/tests/test_warp_codegen.py`
- Segmented one-step `nx=565` profile: one equation-kernel launch per stage; new
  equation/step wall time.
- Headline million-particle fixed-step Warp run (`nx=565`, 10 steps, identical
  physics to the cache-reuse run); compare to 4.51 s and CPU 57.48 s.
- Warp-only adaptive `nx=100` resolved run; compare metrics to the committed
  continuity-density adaptive values (guards the adaptive path against the fp32
  shift in `au,av,aw`).
- `python .ai/.../scripts/validate-memory.py`; `git diff --check`.

## Success criteria

- A generated, cached group kernel runs the continuity half-stage as one
  neighbor-loop launch instead of four, confirmed by a launch/helper-count test
  and the segmented profile.
- `test_warp_codegen.py` passes and the existing focused suite still passes
  (currently 41 passed).
- The million-particle fixed-step Warp wall time improves materially from
  4.51 s / 10 steps.
- CPU/GPU final-state deltas stay near floating-point scale (headline deltas may
  shift from ~1e-8 toward ~1e-6 from fp32 accumulation reordering; reported).
- The adaptive `nx=100` resolved metrics match committed values to fp32 scale.
- No multi-hour run is launched.

## Risks

- Codegen correctness/ergonomics: a malformed snippet yields an opaque Warp
  compile error. Mitigated by validating generated kernels against the
  hand-written oracle and by keeping the snippet vocabulary small and documented.
- `wp.func` resolution from a generated/exec'd kernel namespace is assumed to
  work via the function globals; an explicit early test covers it before the
  full build.
- fp32 accumulation reordering shifts `au,av,aw` ~1e-7 vs the separate kernels;
  bounded by 1e-5 parity tolerances and the order-preserving accumulators.
- Register pressure from one larger kernel could cut occupancy and offset
  traversal savings; the segmented profile and benchmark are the arbiters.
- The adaptive path reads the fused `au,av,aw`; a tiny fp32 shift could perturb
  substep counts over a long run. Guarded by re-validating `nx=100` resolved.
- Transitional duplication (blocks + hand kernels for the same four equations);
  accepted, with migration tracked as an ADR-0003 follow-up.

## Out of scope

- Migrating the per-equation helpers, summation-density, and
  adaptive-timestep-factor paths onto the generator (follow-up slice).
- Gradient/adjoint codegen, multi-group/iterated-group orchestration,
  cross-array (`src != dst`) fused groups.
- Reducing the neighbor-cache build itself (now the largest per-step cost).
- Full `nx=565`, `tf=0.0076` GPU-only elliptical-drop run.

## Estimated effort

One focused implementation/benchmarking session: generator + 4 blocks + tests
+ integration + benchmark (~350–450 LOC across the two host modules and tests).

## Approval

- [x] Plan posted in chat (revised to the generator design after the
  "Generator now" direction was chosen)
- Approved by: @kunalpuri-prediqt at 2026-06-17T18:10:00 CEST
- Approval, verbatim quote:
  > APPROVED
