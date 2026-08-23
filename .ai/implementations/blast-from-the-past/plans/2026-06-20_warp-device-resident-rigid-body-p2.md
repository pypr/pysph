---
type: plan
id: 2026-06-20_warp-device-resident-rigid-body-p2
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-20T06:10:44 CEST
status: approved
aspects: [warp-backend, particle-memory, validation-benchmarks]
adr: ADR-0006
within_boundary: true
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
---

# Plan: Device-resident rigid-body 6-DOF integration (ADR-0006 P2)

## Goal

Keep rigid-body dynamics on the GPU: reduce body moments, finalize force/torque
and angular acceleration, integrate COM linear/angular state with RK2, and
rigid-transform every body particle without a per-stage host copy or host
6-DOF solve. Host transfers occur only when the caller explicitly requests
output or validation data.

## Context

P1 already reduces each body's 16 moment/force/torque terms on the GPU into an
f64 Warp array, but `compute_rigid_body_moments()` immediately copies that tiny
array to NumPy and finalizes the 6-DOF solve on the host. That was selected for
implementation convenience and parity, not because Warp cannot perform the
solve. The user approved changing direction after calling out the architectural
cost: a host finalize introduces synchronization, blocks a fully device-resident
step/CUDA-graph future, and scales poorly with body count.

The production path should therefore remain on-device. The existing
`_rigid_finalize_moments()` stays as a trusted NumPy oracle/debug API. This is an
amendment to ADR-0006's host-finalize choice; its additive-backend, f64-reduction,
and sibling-driver decisions remain unchanged.

## Approach

### 1. Persistent device rigid state

- Add an internal state/buffer object for `nbody` bodies containing f64 device
  arrays for the raw 16-slot reduction, mass, COM, force, acceleration, inertia,
  torque, angular acceleration, COM velocity, angular velocity, and saved RK2
  state. Keep the static `body_id` array on-device.
- Allocate once and reuse across stages/steps. Do not allocate or call
  `.numpy()` in the production stage path.
- Particle properties remain in their configured dtype (normally fp32); compact
  rigid state and reduction stay f64 for stable sums and the 3x3 solve.

### 2. Device finalize and 3x3 solve

- Add a one-thread-per-body Warp kernel that converts the 16 reduced terms into
  COM, parallel-axis inertia, force/COM acceleration, and torque about COM using
  the same formulas as `_rigid_finalize_moments()`.
- Compute `omega_dot = inv(I) * (torque - omega x (I*omega))` on-device using an
  explicit symmetric 3x3 adjugate/determinant solve. Guard non-positive mass and
  singular/near-singular inertia with clear host-entry validation or a device
  error flag checked only at explicit validation/output boundaries.
- Refactor the P1 reduction into a device-returning internal launcher. Preserve
  `compute_rigid_body_moments()` by making it call the device path and pull the
  result only because that public helper explicitly asks for host results.

### 3. Device RK2 integration and rigid transform

- Add additive Warp kernels mirroring `RK2StepRigidBody` and
  `RigidBodyMotion`:
  - initialize/save `x0/y0/z0`, `vc0`, and `omega0`;
  - midpoint stage: `vc = vc0 + 0.5*dt*ac`,
    `omega = omega0 + 0.5*dt*omega_dot`;
  - full stage: `vc = vc0 + dt*ac`,
    `omega = omega0 + dt*omega_dot`;
  - compute each particle velocity as
    `v = vc + omega x (x - cm)` and update stage position from its saved
    position, matching PySPH's RK2 rigid-body semantics.
- Expose a small P2 entry point that performs reduction -> device finalize ->
  RK2 stage -> device motion for synthetic/applied body forces. P3 will compose
  this primitive into `wc_sph_dam_break_rigid_step`; do not modify the existing
  fixed-wall `wc_sph_dam_break_step`.

### 4. Decision and experiment memory

- Amend ADR-0006 to replace host production integration with the device-resident
  design, while retaining `_rigid_finalize_moments()` as the oracle.
- Update the floating-body experiment with P2 commands/results and the explicit
  no-host-round-trip invariant; regenerate the decision graph if ADR frontmatter
  changes.

## Files expected to change

- `pysph/base/warp_sph.py` — persistent rigid state, device finalize/solve,
  RK2/motion kernels, and P2 entry point.
- `pysph/base/tests/test_warp_sph.py` — numerical parity and residency/invariant
  tests.
- `.ai/implementations/blast-from-the-past/decisions/2026-06-19_adr-0006_*.md`
  — record the approved device-resident amendment.
- `.ai/implementations/blast-from-the-past/experiments/2026-06-19-warp-floating-body-rigid/experiment.md`
  — P2 evidence.
- Plan/review/session/current/aspect memory required by the operating contract.

## Tests / validation

- Device finalize parity for asymmetric one- and two-body fixtures, fp32
  particle data and fp64 particle data: mass, COM, inertia, force, acceleration,
  torque, and `omega_dot` match `_rigid_finalize_moments()` at dtype-appropriate
  tolerances.
- RK2 midpoint and full-stage parity against a faithful NumPy implementation of
  `RK2StepRigidBody` + `RigidBodyMotion`, including nonzero translation,
  rotation, off-diagonal inertia, force, and torque.
- Rigid invariants: pairwise body-particle distances remain fixed to numerical
  tolerance; zero force/torque preserves linear/angular velocity; pure
  translation and pure rotation behave correctly; two bodies remain isolated.
- Residency guard: production P2 stepping does not invoke `.numpy()`, ParticleArray
  pull, or host moment finalization inside either stage. One final pull for test
  assertions is allowed.
- Existing P1 host helper tests remain green, proving compatibility.
- Existing 2D generated-source golden/cache guard remains green; no
  `warp_codegen` or shared kernel router edits.
- Focused command:
  `python -m pytest -q pysph/base/tests/test_warp_sph.py -k rigid pysph/base/tests/test_warp_codegen.py::test_2d_path_generated_source_is_byte_identical_to_golden`.
- `validate-memory.py`, `git diff --check`, and an adversarial review before
  requesting @prabhu sign-off.

## Success criteria

- The complete P2 production stage is device-resident and numerically matches
  the NumPy/PySPH formulas.
- No per-stage host synchronization/copy is introduced by rigid integration.
- P1 compatibility and the existing fixed-wall/2D paths remain unchanged.

## Risks

- A nearly singular inertia tensor can destabilize the explicit 3x3 solve;
  reject degenerate bodies and test the error path.
- RK2 semantics can drift if COM/velocity is evaluated from the wrong stage;
  midpoint/full-stage oracle tests pin ordering.
- Updating positions via stage velocities approximates orientation in the same
  way as PySPH's reference step; long-run rigidity drift should be measured in
  P4. A quaternion/orientation-matrix integrator is a later option if the
  reference-compatible path drifts visibly.
- f64 device atomics/math require supported NVIDIA hardware; this was already
  accepted and exercised by P0/P1.

## Out of scope

- Liu fluid/body coupling and NumberDensity pre-pass (P3).
- Collision/contact handling, assembled dam-break case, and rendering (P3/P4).
- CUDA graph capture itself; this slice removes the rigid-body host barrier that
  would prevent it.
- Compyle/Python 3.14 repair and real CPU Application parity.

## Estimated effort

One substantial implementation session, approximately 250-400 LOC across the
backend and focused tests, followed by review.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-20T06:12:47 CEST
- Approval, verbatim quote:
  > approved
