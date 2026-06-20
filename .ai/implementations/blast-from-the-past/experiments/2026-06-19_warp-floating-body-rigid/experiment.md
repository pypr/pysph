---
type: experiment
id: 2026-06-19_warp-floating-body-rigid
created: 2026-06-19T22:45:00 CEST
author: @kunalpuri-prediqt
agent: claude
aspect: validation-benchmarks
adr: ADR-0006
status: active
last_checked: 2026-06-20T13:52:29 CEST
---

# Experiment: Warp floating / rigid body coupled to SPH (3D dam-break)

## Headline

The next Warp benchmark after the 3D Lobovsky dam-break (ADR-0005): a
**floating rigid body** -- the dramatic case is a dam-break surge tossing a
floating box (mirrors `pysph/examples/rigid_body/dam_break3D_sph.py`). The
**deliverable is a photorealistic animation** of the GPU (Warp) simulation,
rendered out-of-band (splashsurf -> Blender Cycles; Omniverse was rejected for
this 8 GB / WSL2 setup -- it can't surface a Warp point cloud natively and its
headless cloud batch is broken).

A fixed wall and a floating body differ by one thing: the body is *integrated*
(6-DOF), the wall is not. So this extends the existing multi-array dam-break
step by lifting "fixed": reduce force+torque over the body, integrate the COM,
and rigid-transform the body particles. All additive (ADR-0006), keeping the 2D
elliptical-drop path byte-identical.

## Validation strategy

The shipped CPU rigid-body Application cannot run on this Python 3.14 venv
(`compyle 0.9.1` uses the removed `ast.Str` for `rigid_body.py`'s matrix
`declare()`s). So validation is against a **faithful numpy reimplementation** of
`RigidBodyMoments`/`Motion` (tier-1 style, as ADR-0005 used for the hand-rolled
CPU baseline). Tier-2 (real CPU Application) is deferred behind a compyle fix.

## Phases (ADR-0006)

- **P0 -- kill-test (DONE).** `p0_rigid_reduce_kill_test.py`: on an asymmetric
  synthetic body (real torque / off-diagonal inertia), prove the one new GPU
  primitive -- an `atomic_add` SUM-reduction producing the 16-slot
  `RigidBodyMoments` `mi` vector -- before touching the backend. Result on the
  RTX 4060:
  - reduction reproduces `RigidBodyMoments`: **2.7e-15** (f64 accum, f32 inputs);
    **8e-14** (f64 data);
  - **fp32 `atomic_add` is non-deterministic: 3.0e-6** run-to-run spread
    (order-dependent / non-associative, unlike the dt-reduce's `atomic_max`);
    **f64 accumulators drop it to ~1e-15** -> locked decision: accumulate in f64;
  - host finalize (COM / inertia tensor / torque-about-COM / `omega_dot`) matches
    numpy to **6e-14**; device rigid-transform exact (**5.5e-17**).

- **P1 -- reduction in the backend (DONE).** Added to `pysph/base/warp_sph.py`,
  all additive (no generated source / router edits):
  - `_rigid_moments_reduce_f32` / `_rigid_moments_reduce_f64` (f64 accumulators
    on both paths), modeled on the `_wcsph_dt_reduce_*` `atomic_max` template;
  - `_rigid_finalize_moments` (host numpy 6-DOF moments, reusing
    `rigid_body.py:128-207`);
  - `compute_rigid_body_moments(pa, nbody, omega, ...)` driver.
  Tests (`pysph/base/tests/test_warp_sph.py`):
  `test_rigid_body_moments_matches_reference_3d[False/True]` (two-body,
  asymmetric; f32 and f64 paths vs the numpy reference) and
  `test_rigid_moments_f32_kernel_is_accurate_and_deterministic`. The
  cache-stability guard `test_2d_path_generated_source_is_byte_identical_to_golden`
  still passes.

- **P2 (DONE, amended to fully device-resident).** The implementation owner
  rejected a host-side production solve because the goal is a genuinely GPU-
  accelerated backend. Added persistent `WarpRigidBodyState` plus additive Warp
  kernels for moment finalize, the symmetric 3x3 angular-acceleration solve,
  RK2 saved/midpoint/full compact state, rigid velocity
  `v = vc + omega x (x-cm)`, and stage position updates. The production stage
  performs no host finalize, pull, `.numpy()`, or explicit synchronization;
  `_rigid_finalize_moments` is retained only as the validation/host-query oracle.
  Device finalize and both RK2 stages match faithful NumPy/PySPH formulas for
  asymmetric two-body fp32/fp64 cases; pure translation preserves body geometry
  and bodies remain isolated. Focused P1/P2 + 2D cache guard: `10 passed`;
  final full Warp SPH regression: `49 passed`.
- **P3 (next).** Liu coupling group + `NumberDensity` pre-pass + sibling driver
  `wc_sph_dam_break_rigid_step` (body excluded from the PEC stage); resolve the
  `arho` double-count.
- **P4.** EPEC fidelity + existing-driver behaviour guard; assemble the 3D
  surge-tosses-a-box case; render.

## P2 runtime case (2026-06-20)

Ran an asymmetric 3D box (9x7x5 = 315 particles) under prescribed nonzero net
force and torque for 2,000 RK2 steps at `dt=1e-4` (`t=0.2`) on the RTX 4060.
The stepping loop called only `save_rigid_body_state` and the two device RK2
stages; host arrays were read once after the final synchronization.

```text
wall_s:                       0.5445955659997708
steps_per_s:                  3672.4500250537144
particle_steps_per_s:         1156821.7578919202
device_error:                 0
all_finite:                   true
vc_max_abs_error:             1.552180384223334e-09
initial_omega:                [0.2, -0.1, 0.3]
final_omega:                  [0.4878298261, -0.1616930311, 0.4444991226]
max_pair_distance_drift:      4.1726284255583224e-07
relative_pair_distance_drift: 9.386771416218177e-07
```

This validates repeated GPU-resident rigid stepping and expected force/torque
response. It is not a fluid-coupled floating-body claim; Liu coupling is P3.

## How to run

```bash
PY=/home/kunalp/.pqt_venv_e0b41259/bin/python
# P0 kill-test (standalone, no backend dependency beyond warp)
$PY .ai/implementations/blast-from-the-past/experiments/2026-06-19_warp-floating-body-rigid/p0_rigid_reduce_kill_test.py
# P1 backend tests + cache guard
$PY -m pytest -q pysph/base/tests/test_warp_sph.py -k rigid \
   pysph/base/tests/test_warp_codegen.py::test_2d_path_generated_source_is_byte_identical_to_golden
```

## Out of scope (follow-ups)

- The fluid<->body coupling physics + collision (P3); the assembled 3D animation
  case; tier-2 CPU parity (needs the compyle py3.14 fix).
