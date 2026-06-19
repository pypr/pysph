---
type: plan
id: 2026-06-18_warp-3d-dam-break-lobovsky-benchmark
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-18T20:45:00 CEST
status: approved
aspects: [warp-backend, gpu-nnps, validation-benchmarks]
adr: ADR-0005
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
within_boundary: true
---

# Plan: Warp 3D dam-break Lobovsky benchmark

## Goal

Add a 3D dam-break benchmark (PySPH Lobovsky no-obstacle case first) to the Warp
backend, validated against the shipped CPU reference. Implement the four WCSPH
features the 2D elliptical drop never exercised -- gravity, multi-array stepping,
solid walls, and the WendlandQuintic kernel -- as **purely additive** extensions
(ADR-0005) so the 2D elliptical-drop path and its on-disk Warp kernel cache stay
**byte-identical**.

## Context

A 5-area investigation confirmed the lower layers are already fully 3D and
3D-validated: `warp_nnps.py` bins on `(ix,iy,iz)` with `cid = ix + iy*nx +
iz*nx*ny` and a 27-cell `range(-1,2)^3` walk (`:249-376`, `:983-997`);
`warp_codegen.py` emits `dx/dy/dz`, `vijx/vijy/vijz`, a 3D triple cell loop and
3D `cid` under `dim` guards (`:222-261`, `:368-405`); `warp_sph.py` carries
`dim==3` kernel normalization and `z`/`w` integration in every step path.
Existing dim=3 parity tests pass (`test_warp_sph.py:361,512,604`;
`test_warp_nnps.py:252,286`).

Confirmed-absent (grep) and needed by a dam-break: no gravity term anywhere; all
drivers are single-array (`src==dst`, `wc_sph_leapfrog_step`/
`_wc_sph_pec_continuity_step` at warp_sph.py:1888/1840); no fixed-wall handling
or `TaitEOSHGCorrection`; `_kernel_id` (warp_sph.py:1066) knows only
cubic(0)/gaussian(1) while both CPU references use `WendlandQuintic(dim=3)`. The
2D-ness otherwise lives only in the elliptical-drop runner (`dim=2`, 2D disk IC,
2D radius metric).

CPU reference: `pysph/examples/dam_break/dam_break_3d_lobovsky.py` (container
~5.367 x 0.5 x 1.5, fluid column 2.0 x 0.5 x 1.0, `dx=H/30`, `hdx=1.3`,
`rho0=1000`, `gamma=7`, `alpha=0.25`, `beta=0`, `gz=-9.81`, `c0~73`,
`WendlandQuintic(dim=3)`, `tf=2.5`, `n_damp=50`); geometry via
`DamBreak3DGeometry` (`pysph/examples/_db_geometry.py:250-432`), scheme
`WCSPHScheme` + `EPECIntegrator`/`WCSPHStep`.

## Approach

Sequence: pin 3D tests -> Wendland -> gravity -> multi-array/walls -> runner ->
validate & record. Each backend addition is additive; ids 0/1, existing emitted
source, and single-array drivers are never edited.

1. **Pin 3D coverage first (no production change).** Add a dim=3 codegen grid
   test (a `_KernelSum`-style grid kernel with `nz>1`, real z coords) to
   `test_warp_codegen.py`, and dim=3 variants of the fused-accel, grid-direct,
   and adaptive-timestep parity tests to `test_warp_sph.py` (the `_cpu_wcsph_dt`
   helper already handles dim=3). Locks the "2D byte-identical" invariant as
   enforceable before any edit.
2. **WendlandQuintic (`warp_sph.py`).** Add `_wendland_quintic_*` value + dwdq
   device funcs with `dim==1/2/3` normalization, mirror the cubic/gaussian
   structure (warp_sph.py:16-280), route through `_kernel_value_*`/`_kernel_dwdq_*`,
   and extend `_kernel_id` (`:1066`) with a new id for `'wendland'`. Ids 0/1
   unchanged. Add a dim=3 CPU-parity test vs `pysph.base.kernels.WendlandQuintic`.
3. **Gravity (`warp_sph.py`).** Add `_apply_body_force_*` (`u+=gx*dt`;
   `v+=gy*dt` if `dim>1`; `w+=gz*dt` if `dim>2`) and a host driver
   `apply_body_force(pa, gx, gy, gz, dt, dim, ramp)`. Thread optional
   `gx=gy=gz=0` + `n_damp` ramp into the dam-break step (not the elliptical-drop
   step). Never fold `g` into generated equation source. Add an exact-arithmetic
   gravity test.
4. **Multi-array + solid walls (`warp_sph.py`).** Add a `wc_sph_dam_break_step`
   that, per destination array, sums equation-group contributions over a source
   list (fluid from `[fluid, wall]`, wall `drho/dt` from `[fluid]`) using the
   already-`src!=dst` groups; integrate only fluid positions/velocities (walls
   fixed). Add a `_tait_eos_hg_correction_*` device kernel (clamp `p>=0`,
   recompute `rho`) for solid arrays. Add a dim=3 two-array integration smoke
   test (all_finite over a few steps) and a two-array dim=3 NNPS test.
5. **Runner experiment packet.** New dir
   `experiments/2026-06-18_warp-dam-break-3d-runner/` mirroring the
   elliptical-drop packet: `dam_break_3d_runner.py` (Runner class; IC via
   `DamBreak3DGeometry` or a replica; `UniformGridWarpNNPS(dim=3, [fluid,wall])`;
   gravity + adaptive dt + `n_damp`; `_metrics` adds z-extent/max-height/3D KE),
   `run_correctness.sh` smoke wrapper, a tier-1 `compare_warp_pysph_dam_break_3d.py`
   (hand-rolled CPU equations + `LinkedListNNPS(dim=3)` + Wendland), and a tier-2
   `resolved_dam_break_3d_comparison.py` (subprocess the real PySPH Application,
   load via `pysph.solver.utils.load`, step Warp to matched checkpoints).

## Files expected to change

- `pysph/base/warp_sph.py` (Wendland, body force, HG correction, multi-array
  dam-break step) -- additive only
- `pysph/base/tests/test_warp_sph.py`, `pysph/base/tests/test_warp_codegen.py`
  (dim=3 + new-kernel parity tests)
- New `experiments/2026-06-18_warp-dam-break-3d-runner/` (runner, smoke wrapper,
  tier-1/tier-2 comparison scripts, experiment.md, per-run summary JSON)
- ADR-0005 (registered), aspect contexts (warp-backend, validation-benchmarks),
  current.md, daily, session log, review artifact

## Tests / validation

Three tiers, all gating on `all_finite` and a stable step count:

- **Unit/parity (pytest):** dim=3 codegen 27-cell grid path; dim=3 fused /
  grid-direct WCSPH accel; dim=3 adaptive timestep; WendlandQuintic value+dwdq vs
  `pysph.base.kernels.WendlandQuintic`; the gravity kernel (exact `g*dt`); the
  `TaitEOSHGCorrection` solid kernel; a dim=3 two-array (fluid+wall) integration
  smoke. **Regression guard:** assert the cubic/gaussian summation kernel source
  string is byte-identical before/after (2D path unchanged).
- **Tier-1 smoke parity:** hand-rolled CPU reimplementation (continuity, Tait +
  HG on walls, pressure gradient, Monaghan AV, XSPH, gravity, adaptive dt, PEC
  ordering) with `LinkedListNNPS(dim=3)` + Wendland on a small grid, compared
  field-by-field to Warp.
- **Tier-2 resolved parity:** subprocess `dam_break_3d_lobovsky.py`, load output,
  step Warp to matched checkpoint times; report per-checkpoint CPU-vs-Warp signed
  deltas. Observables: per-particle x/u/v/w/rho/p at short horizon; global KE;
  surge-front x-position vs time; max height; probe-point pressure
  (`p/(rho g H)` vs `t sqrt(g/H)`) vs `db_exp_data.get_lobovsky_data()`.
- `validate-memory.py`; `git diff --check`.

## Risks

- **Disk-cache stability (top constraint):** any incidental edit to existing
  generated source busts the 2D cache. Mitigated by additive-only design (new id,
  separate gravity kernel default `g=0`, separate driver) + the byte-identical
  source regression test.
- **Gravity placement / `n_damp` ramp:** must match the CPU
  `MomentumEquation`+`WCSPHStep` ordering or the free surface evolves wrong while
  passing `all_finite`. Guarded by tier-1 parity at short horizon.
- **Wall fidelity:** `TaitEOSHGCorrection` + fixed-particle handling is the crux
  of wall repulsion; a single wall layer can leak if `h/dx`/`radius_scale` differ
  from the reference. Match `dx`, `h=hdx*dx`, `m=rho0*dx^3` exactly.
- **EPEC vs PEC:** CPU ref uses EPEC, Warp uses PEC. Either run the CPU ref with
  `PECIntegrator` for strict parity or document the difference. To be decided in
  step 5.
- **fp32 vs fp64 + chaos:** long-horizon per-particle parity is meaningless;
  rely on aggregate/experimental observables for the headline.
- **3D memory/perf:** `ncells = nx*ny*nz` grows cubically; per-update host
  readback of x/y/z/h is heavier in 3D. Correctness-only here; profile before any
  speed claim. (Not optimizing the 27-cell `dzc` loop -- would change the 2D
  path.)

## Out of scope

- SPHERIC/Kleefsman obstacle case (third array); a cubic/gaussian dam-break
  variant; performance/cross-GPU characterization (separate follow-up).
- Any edit to existing kernel ids 0/1, generated equation source, or the
  single-array elliptical-drop step.
- Non-cubic periodic boxes; narrowing the flat host-query path.

## Estimated effort

Large overall, unevenly distributed: NNPS/codegen 3D = zero code (additive tests
only); `warp_sph.py` physics features = the bulk of production code (medium-large,
all additive); runner + IC + two-tier harness = medium (reuses scaffolding);
tests = medium. Ship Lobovsky no-obstacle before SPHERIC.

## Approval

- [x] Plan posted in chat and approved
- Approved by: @kunalpuri-prediqt at 2026-06-18T20:50:00 CEST
- Approval, verbatim quote:
  > approved
