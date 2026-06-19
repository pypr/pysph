---
type: decision
id: ADR-0005
date: 2026-06-18
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0003, ADR-0004]
depends_on: [ADR-0003]
conflicts_with: []
---

# ADR-0005: Adopt additive 3D WCSPH physics for the Warp dam-break benchmark

## Context

The Warp backend has one validation case: a 2D free-surface elliptical drop. We
want a second, well-known 3D case -- the dam-break (Lobovsky no-obstacle first,
SPHERIC/Kleefsman obstacle case second) -- to strengthen the implementation and
the upstream PR.

A 5-area code investigation found that the **lower layers are already fully
3D-capable** and 3D-validated by existing tests, with no changes required:

- NNPS (`warp_nnps.py`): 3D cell binning (`cid = ix + iy*nx + iz*nx*ny`), a true
  27-cell `range(-1,2)^3` walk, `nz`/`zmin`/ghost-padding under `dim>2`, a 3D
  periodic box. Tests `test_warp_nnps.py:252,286` already exercise dim=3.
- Codegen (`warp_codegen.py`): emits `dx/dy/dz`, `vijx/vijy/vijz`, the 3D triple
  cell loop and 3D `cid`, and per-dim periodic min-image -- all `dim`-gated.
- Physics (`warp_sph.py`): Gaussian/cubic kernel normalization carries the
  correct `dim==3` constants (`pi^-3/2 / h^3`, `1/(pi h^3)`); every step path
  (Euler/KDK/continuity-PEC) reads/writes `z` and `w` under `dim>2` guards.
  Per-equation device kernels have passing dim=3 cross-array parity tests
  (`test_warp_sph.py:361,512,604`).

What a dam-break needs that the elliptical drop never exercised (confirmed
absent by grep):

1. **Body force / gravity** -- there is no `gx/gy/gz` term anywhere; without
   `gz=-9.81` the dam-break is identically static.
2. **Multi-array stepping** -- every driver passes `src==dst` (single array).
   A dam-break needs fluid accelerations summed over `[fluid, wall(, obstacle)]`
   and wall density from `[fluid]`.
3. **Solid walls** -- fixed (non-integrated) boundary particles whose pressure is
   enforced via `TaitEOSHGCorrection` (clamp `p>=0`, recompute `rho` from `p`).
   No such concept exists today.
4. **WendlandQuintic kernel** -- both CPU references use `WendlandQuintic(dim=3)`,
   but `_kernel_id` (warp_sph.py:1066) only knows cubic(0)/gaussian(1).

The hard constraint is **disk-cache stability**: Warp compiles generated kernels
and caches them on disk keyed by a deterministic md5 of the kernel source
(established when the order-dependent name bug was fixed). Any incidental edit to
an *existing* generated equation block's source string would silently force a
recompile and could perturb the committed 2D elliptical-drop baseline.

## Decision

Add the four missing WCSPH features as **purely additive extensions**, so that
the 2D elliptical-drop code path and its on-disk kernel cache remain
**byte-identical**:

1. **WendlandQuintic as a new kernel id.** Add `value` + `dwdq` device functions
   with correct `dim==1/2/3` normalization and map `'wendland'` to a new id.
   Leave ids 0/1 (cubic/gaussian) and their emitted source unchanged.
2. **Gravity as a standalone additive velocity kernel** (`_apply_body_force_*`:
   `u += gx*dt` always, `v += gy*dt` under `dim>1`, `w += gz*dt` under `dim>2`),
   with a linear ramp over `n_damp` steps to match the CPU reference startup
   (`n_damp=50`). Defaults `gx=gy=gz=0`. It is applied in the step driver, never
   folded into `PressureGradient`/`ContinuityEquation`/the PEC-stage emitted
   source. With `g=0` the existing step is unchanged.
3. **Multi-array (fluid + solid) stepping.** Add a driver that, per destination
   array, sums neighbor contributions over a list of source arrays, reusing the
   already-`src!=dst`-capable equation groups and NNPS query (no codegen/NNPS
   change). Solid arrays are *fixed*: their `rho`/`p` update but their position
   and velocity are not integrated.
4. **Solid pressure via a new `TaitEOSHGCorrection` device kernel** (clamp
   `p>=0`, recompute `rho` from `p`) applied to solid arrays only, mirroring
   `pysph/sph/wc/basic.py` / `scheme.py:422-426`.

The 2D path keeps `g=0`, single-array stepping, ids 0/1, and no HG correction.
The precise invariant (verified, see Validation): the **generated group source**
for the cubic/gaussian 2D kernels is **byte-identical** (the kernel choice is a
runtime `kernel_id`, not source), enforced by
`test_2d_path_generated_source_is_byte_identical_to_golden`, and the cubic
(id 0) / gaussian (id 1) **numerical code paths** in the shared
`_kernel_value`/`_kernel_dwdq` routers are **unchanged** (only an additive
`if kernel_id == 2` branch was added). The router *device-function source* did
grow by that branch, so Warp's on-disk module hash for kernels that call the
router changes -- a one-time, **logic-preserving recompile** of the warm cache
can occur. It does not change the cubic/gaussian results and does not perturb
the committed 2D elliptical-drop baseline. (Earlier drafts said "on-disk cache
byte-identical"; the accurate statement is "emitted source byte-identical +
logic-preserving router extension".)

The first target is the **Lobovsky no-obstacle** case (fluid + one wall array,
`tf=2.5 s`), then the SPHERIC obstacle case (three arrays, `tf=6.0 s`).

## Rationale

- Additive-only is the smallest correct change that satisfies the
  disk-cache-stability constraint: new kernel id, separate gravity kernel,
  separate multi-array driver, new solid-EOS kernel -- none edit existing
  generated source.
- The expensive infrastructure (3D NNPS, 3D codegen, dim-correct kernels, 3D
  step paths) is already present and validated, so the work is confined to the
  physics features the 2D drop never needed plus a runner -- not a dimensional
  rewrite.
- Cross-array per-equation kernels are already 3D-validated, so multi-array
  stepping is driver wiring plus one solid-EOS kernel, not new neighbor-loop code.

## Alternatives considered

- **Fold gravity into the generated momentum/PEC source.** Rejected: it mutates
  an existing generated kernel's source string, busting the 2D disk cache and
  risking a perturbed baseline. A standalone `g`-kernel keeps the contract.
- **Reuse cubic/gaussian instead of adding Wendland.** Rejected for the parity
  case: the CPU references use `WendlandQuintic(dim=3)`; matching the kernel is
  needed for an honest CPU-vs-Warp comparison. (A cubic/gaussian dam-break could
  be run as an extra, but the headline parity uses Wendland.)
- **Generalize the single-array step in place** rather than add a new driver.
  Rejected: it would change the existing step's launch wiring and risk the 2D
  path; a separate multi-array driver leaves the single-array path intact.
- **EPEC integrator parity.** The CPU references use `EPECIntegrator`; the Warp
  backend implements a single PEC stage. Either run the CPU reference with
  `PECIntegrator` for strict per-particle parity, or document EPEC-vs-PEC as a
  known small difference. Decision deferred to the plan; aggregate observables
  (KE, surge front, probe pressure vs experiment) are the primary validators
  because fp32-vs-fp64 chaotic flow defeats long-horizon per-particle parity.

## Consequences

- New device functions and a kernel id in `warp_sph.py` (Wendland, body force,
  HG correction) and a new multi-array dam-break step driver; all additive.
- A 3D dam-break experiment packet (runner + initial condition via
  `DamBreak3DGeometry` + two-tier CPU-parity harness) and dim=3 tests for the
  fused/grid-direct/adaptive paths plus the new kernels.
- The 2D elliptical-drop path keeps its kernel ids and its *generated source*
  byte-identical (test-asserted); the cubic/gaussian numerical results are
  unchanged. The shared kernel-id router gained an additive `id==2` branch, so a
  one-time logic-preserving recompile of the on-disk cache can occur without
  changing results.
- Validation shifts from an analytic locus (elliptical drop) to **aggregate /
  experimental observables** (total KE, surge-front position, max height,
  probe-point pressure vs `db_exp_data` Lobovsky/Kleefsman data), since the 3D
  dam-break has no closed-form solution and fp32-vs-fp64 trajectories diverge.

## Follow-ups

- SPHERIC/Kleefsman obstacle case (third particle array) after Lobovsky lands.
- Optional: a cubic/gaussian dam-break variant for an additional cross-check.
- Performance characterization (per-step wall, cross-GPU) once correctness is
  established -- separate from this correctness-focused ADR.
- Revisit whether the 27-cell stencil should special-case the `dzc` loop on
  `dim` for 2D speed (explicitly out of scope here to keep the 2D path
  byte-identical).

## Validation (2026-06-19)

Implemented and validated; experiment packet
`experiments/2026-06-18_warp-dam-break-3d-runner/`.

- **Backend unit/parity (pytest):** the new-feature tests pass --
  WendlandQuintic(dim=3) summation density, ramped body force (+ 2D guard),
  `TaitEOSHGCorrection`, and the two-array 3D dam-break step (walls fixed, all
  finite, gravity drives the fluid down); plus the dim=3 codegen-grid,
  fused/grid-direct, and adaptive-timestep pins; plus the new byte-identical
  source guard.
- **Tier-1 (hand-rolled CPU EPEC):** fluid x/y/z/u/v/w + fluid/wall rho match
  Warp (fp32) to ~1e-8 relative; pressure matches in absolute terms at the fp32
  Tait-EOS cancellation floor.
- **Tier-2 (real `dam_break_3d_lobovsky.py` Application):** at matched
  checkpoints, KE / surge-front / max-height / density agree to fp32 scale;
  pressure is ~13% relative near rest (fp32 floor) and recovers to ~1% relative
  once the column collapse develops a real pressure field.

Resolved deferrals from this ADR:

- **EPEC vs PEC** (Alternatives): `wc_sph_dam_break_step` is E-P-E-C (re-evaluates
  before the predictor), matching the reference `EPECIntegrator`. No
  `PECIntegrator` substitution is needed; the reference runs with its native
  integrator.
- **Gravity ramp vs `n_damp`** (Decision step 2): PySPH `n_damp` damps the
  *timestep* (`0.5*(sin(pi*(-0.5+(count+1)/n_damp))+1)`), not gravity. The runner
  matches PySPH by applying that factor to the adaptive `dt` (full gravity); the
  additive `gravity_ramp` feature exists but is unused for this benchmark.
- **`c0`:** matches the reference scheme constant `10*sqrt(2*9.81*0.55) ~ 32.85`
  (its initial-`dt` `get_max_speed` form `~44.29` is used only to cap `dt`).
