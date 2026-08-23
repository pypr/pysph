---
type: decision
id: ADR-0006
date: 2026-06-19
author: @kunalpuri-prediqt
scope: warp-backend
status: Accepted
supersedes: []
relates_to: [ADR-0003, ADR-0004, ADR-0005]
depends_on: [ADR-0005]
conflicts_with: []
---

# ADR-0006: Adopt additive rigid-body coupling for the Warp floating-body benchmark

## Context

ADR-0005 gave the Warp backend a 3D dam-break (Lobovsky no-obstacle): fluid +
fixed solid walls. The next benchmark is a **floating / rigid body** -- the
dramatic 3D dam-break surge tossing a floating box (mirrors
`pysph/examples/rigid_body/dam_break3D_sph.py`) -- and the **deliverable is a
photorealistic animation** of a GPU (Warp) simulation. The animation is rendered
out-of-band (splashsurf -> Blender Cycles); this ADR is only about producing the
GPU simulation data.

A fixed wall and a floating body differ by exactly one thing: the wall is *not
integrated* (the shared PEC stage leaves it in place while its density/pressure
respond to the fluid), whereas a floating body **moves under integrated 6-DOF
rigid-body dynamics**. So the floating body is the existing multi-array
dam-break path with "fixed" lifted: sum the fluid force + torque over the body's
particles, integrate the body's centre-of-mass linear + angular momentum, and
rebuild each body particle's position/velocity from the rigid transform.
Buoyancy is then emergent from the pressure field -- there is no explicit
buoyancy term.

PySPH ships the reference machinery in `pysph/sph/rigid_body.py`
(`RigidBodyMoments`, `RigidBodyMotion`, `LiuFluidForce`/`AkinciRigidFluidCoupling`,
`RK2StepRigidBody`). Two constraints shape the design:

1. **Disk-cache stability (inherited from ADR-0005).** Any edit to an existing
   *generated* equation block's source string, or to the shared kernel-id
   router source, risks busting the 2D elliptical-drop on-disk cache /
   perturbing the committed baseline. The new work must be **purely additive**.
2. **No runnable CPU reference on this env.** `rigid_body.py`'s matrix
   `declare()` statements hit `compyle 0.9.1`'s use of the Python-3.12-removed
   `ast.Str` on the Python 3.14 venv, so the shipped CPU rigid-body Application
   does not run here. Validation is therefore against a **faithful numpy
   reimplementation** of `RigidBodyMoments`/`Motion` (the tier-1 style ADR-0005
   already used), with the full CPU Application (tier-2) deferred behind a
   compyle upgrade.

A P0 kill-test (`p0_rigid_reduce_kill_test.py`) de-risked the one genuinely new
GPU primitive before any backend edit: on an asymmetric synthetic body, an
`atomic_add` sum-reduction reproduced the 16-slot `RigidBodyMoments` `mi` vector
exactly, the host finalize matched numpy to ~6e-14, and the device transform was
exact -- but **fp32 `atomic_add` is non-deterministic** (~3e-6 run-to-run
spread, because it is order-dependent / non-associative, unlike the
order-independent `atomic_max` of the dt-reduce). Accumulating in f64 collapses
the spread to f64 round-off (~1e-16).

## Decision

Add rigid-body dynamics as **purely additive extensions**, so the 2D
elliptical-drop path and its on-disk kernel cache stay byte-identical (verified
by `test_2d_path_generated_source_is_byte_identical_to_golden`, which still
passes):

1. **Force/torque SUM-reduction on the device** (`_rigid_moments_reduce_f32`/
   `_rigid_moments_reduce_f64`): per body, `atomic_add` the 16-slot `mi` vector
   of `RigidBodyMoments.reduce` (total mass; `m*x/y/z` for the COM; six
   second-moments/products of inertia about the ORIGIN; total force; torque
   about the origin). **Accumulate in f64 on both paths** -- the f32 kernel
   casts f32 particle data to f64 before accumulating. Standalone launch
   kernels modeled on the existing `_wcsph_dt_reduce_*` `atomic_max` template;
   they touch no generated source and no kernel-id router.
2. **Host (numpy) 6-DOF finalize** (`_rigid_finalize_moments`): COM,
   parallel-axis inertia tensor about the COM, total force, COM acceleration,
   torque about the COM, and `omega_dot = inv(I)(tau - omega x (I omega))`,
   reusing PySPH's exact formulas (`rigid_body.py:128-207`). At one (few) body
   the device->host round-trip is free (the backend already syncs every launch
   and copies the dt scalar to host each step), and it reuses PySPH's verified
   math rather than reimplementing an inertia-tensor inverse / quaternion update
   in a kernel.
3. **Device rigid-transform** of body particle positions/velocities from the
   integrated COM state (`v = vc + omega x r`, `x += ...`), validated exact in
   P0; to be wired in P2.
4. **A new sibling step driver** `wc_sph_dam_break_rigid_step` (P3): the body is
   **excluded from the shared PEC launch entirely** and advanced by the rigid
   transform instead. `wc_sph_dam_break_step` (the fixed-wall path) is **never
   edited**.
5. **Fluid<->body coupling as its own additive `WarpEquation` group** (P3): the
   *real* reference coupling (`LiuFluidForce` -- a standard symmetric momentum
   term with a clean equal-and-opposite body reaction), with a `NumberDensity`
   volume pre-pass for the density path. Not a "wall pressure-gradient mirror":
   validating against PySPH-Liu requires implementing Liu, and the equilibrium
   draft alone is coupling-blind.

### Amendment — 2026-06-20: production 6-DOF stays on the GPU

The original P1/P2 split selected a host NumPy finalize and host integration
because one body's 3x3 solve is tiny and easy to validate. The implementation
owner challenged the architectural cost: even a tiny host solve creates a
synchronization/copy boundary, prevents a fully device-resident step and future
CUDA-graph capture, and scales poorly with body count. The direction change was
approved explicitly before the P2 plan.

For the production path, decision items 2 and 3 are therefore amended:

- the f64 16-slot reduction remains on-device and feeds a one-thread-per-body
  Warp finalize kernel;
- mass, COM, inertia, force, acceleration, torque, angular acceleration, COM
  velocity, angular velocity, and saved RK2 state live in persistent f64 device
  arrays;
- the symmetric 3x3 solve for
  `omega_dot = inv(I)(tau - omega x (I omega))`, RK2 midpoint/full body-state
  updates, `v = vc + omega x r`, and particle position updates run in Warp;
- the production RK2 stage performs no `.numpy()`, ParticleArray pull, or
  explicit device synchronization; host transfer is reserved for checkpoints
  and validation;
- `_rigid_finalize_moments()` remains as the trusted NumPy oracle and explicit
  host-result helper, not the stepping implementation.

The additive/cache-stability, f64 reduction, Liu coupling, and sibling-driver
parts of this ADR are unchanged.

## Rationale

- The only new device primitive is a hand-written `atomic_add` reduction with
  its own structural cache key; everything else is host numpy or an additive
  equation group / driver. Nothing edits an existing generated block's source,
  so the cache-stability contract holds (golden md5 test green).
- f64 accumulators are the cheap fix for the fp32 `atomic_add` non-associativity
  the adversarial review flagged: ~6 orders tighter run-to-run, negligible cost
  at body-particle counts.
- The standalone rigid kernels keep the project's "byte-identical 2D path"
  mental model intact; the NumPy implementation remains the parity oracle for
  the device 6-DOF math.

## Alternatives considered

- **Integrate 6-DOF on the device** (inertia-tensor inverse / quaternion in a
  kernel). Rejected for now: at one body it buys nothing, adds fp32-atomic and
  reimplementation risk, and complicates parity. Revisit only if many bodies.
- **f32 accumulators.** Rejected: order-dependent, ~3e-6 non-deterministic
  (P0) -- breaks reproducibility for negligible saving.
- **Reuse the fused fluid momentum block with the body as destination** for the
  coupling. Risk: the fused block writes continuity (`arho`) *and* momentum
  together, so body density would be double-counted (also fed by the wall-style
  continuity pass) -> wrong pressure -> wrong force. To be checked at P3; if
  present, use a momentum-only generated *variant* (additive, new structural
  cache key) rather than editing the fused block.
- **Mirror the wall pressure-gradient path as the body coupling.** Rejected: the
  reference uses Liu; validating a different force model against PySPH-Liu is
  uninterpretable, and equilibrium draft cannot catch the error.

## Consequences

- New standalone kernels (`_rigid_moments_reduce_f32/f64`, device finalize,
  RK2 state, and rigid motion) plus persistent compact device state in
  `warp_sph.py`; later a coupling group and sibling driver. All additive; the
  single-array and fixed-wall paths are untouched.
- The 2D elliptical-drop generated source stays byte-identical (test-asserted);
  the new kernels are not in `_WARP_DEVICE_FUNCS` and not seeded into generated
  kernels, so the existing kernels' disk-cache hashes are unchanged.
- Validation is against a numpy `RigidBodyMoments`/`Motion` reimplementation
  (tier-1); tier-2 vs the real CPU Application needs a compyle py3.14 fix.
- fp32 rigid reduction is ~deterministic (f64 round-off) but not bit-identical,
  a small, documented relaxation of the "byte-identical" mental model for the
  rigid path only.

## Follow-ups

- P2: device-resident 6-DOF finalize/integrate + device transform wired
  together. (Completed 2026-06-20; see validation below.)
- P3: Liu coupling group + `NumberDensity` pre-pass + `wc_sph_dam_break_rigid_step`;
  resolve the `arho` double-count; bolt-on `RigidBodyWallCollision` if the case
  needs it.
- P4: EPEC fidelity (rigid update placement across both half-stages) + a
  behaviour-stability guard for the existing driver.
- Assemble the 3D surge-tosses-a-box case; render via splashsurf -> Blender.
- Optional: compyle upgrade for tier-2 CPU parity.

## Validation (2026-06-19)

- **P0 kill-test** (`p0_rigid_reduce_kill_test.py`, RTX 4060): `atomic_add`
  reduction reproduces `RigidBodyMoments` (2.7e-15 given f32 inputs; 8e-14 for
  f64 data), host finalize matches numpy to 6e-14, device transform exact
  (5.5e-17); fp32 accum spread 3.0e-6 vs f64 accum 1.4e-15 -> **decision: f64
  accumulators**.
- **P1 (this change):** `_rigid_moments_reduce_f32/f64` + `_rigid_finalize_moments`
  + `compute_rigid_body_moments` in `warp_sph.py`, with tests
  `test_rigid_body_moments_matches_reference_3d[False/True]` (two-body,
  asymmetric; f32 and f64 paths match the numpy reference) and
  `test_rigid_moments_f32_kernel_is_accurate_and_deterministic`. The golden
  2D-source guard still passes (cache-stability preserved).
- **P2 (2026-06-20 amendment):** persistent `WarpRigidBodyState`, device
  finalize/symmetric-3x3 solve, device RK2 midpoint/full updates, and device
  `RigidBodyMotion` equivalent. fp32/fp64 device results match the NumPy/PySPH
  formulas; a guard monkeypatches host finalize, ParticleArray pull, and
  `wp.synchronize_device` to fail if either production stage crosses the host
  boundary. Focused rigid + 2D cache guard: `10 passed`; final full Warp SPH
  regression: `49 passed`.
- **P3 (2026-06-20):** additive `RigidNumberDensity` plus deterministic
  two-pass `LiuFluidAcceleration`/`LiuBodyReaction`, rigid density/body-force
  staging, and sibling `wc_sph_dam_break_rigid_step`. The rigid body is excluded
  from the pre-existing fused fluid block and contributes to fluid/body
  continuity exactly once in each direction. fp32/fp64 Liu results match a
  direct NumPy/Wendland reference with equal total reaction; the final suite is
  `54 passed`. A 7,458-particle, 241-step first-plunge transient remained finite,
  moved and rotated from computed fluid force, and preserved relative body
  geometry to `1.90e-6`. Review:
  `reviews/2026-06-20_warp-liu-fluid-rigid-coupling-p3.md`.
