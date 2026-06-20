---
type: review
date: 2026-06-20
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-06-20_warp-device-resident-rigid-body-p2.md
adrs: [ADR-0006]
aspects_touched: [warp-backend, particle-memory, validation-benchmarks]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Device-resident rigid-body 6-DOF integration (P2)

## Diff summary

- Added persistent `WarpRigidBodyState` device buffers for one or more bodies:
  f64 moment reduction, mass/COM/inertia/force/torque/acceleration, linear and
  angular state, and RK2 saved state; static body IDs stay on-device.
- Added a one-thread-per-body Warp finalize kernel with an explicit symmetric
  3x3 solve for angular acceleration and device error flags.
- Added device kernels/API for start-state save, RK2 midpoint/full compact-state
  updates, rigid velocity `vc + omega x r`, and stage position updates.
- Refactored the P1 reduction into a reusable no-sync launcher. The old
  `compute_rigid_body_moments()` host-result API is preserved; production P2
  uses `compute_rigid_body_moments_device()` and performs no host transfer.
- Added setup-time empty/zero-mass/singular-inertia rejection.
- Added fp32/fp64 device-finalize and RK2 parity tests, a no-host-barrier guard,
  pure-translation geometry/multi-body checks, and singular-geometry coverage.
- Amended ADR-0006 and the floating-body experiment to record the explicitly
  approved GPU-resident direction.

## Aspects touched and host files modified

- `warp-backend`: standalone Warp kernels and orchestration API.
- `particle-memory`: compact rigid state remains device-authoritative; host
  copies only at explicit output/validation boundaries.
- `validation-benchmarks`: faithful NumPy/PySPH oracle parity and architectural
  no-host-barrier gate.
- Host files: `pysph/base/warp_sph.py`,
  `pysph/base/tests/test_warp_sph.py` — both within the approved boundary.

## Behavioral / numerical changes

- New behavior is additive; no existing WCSPH/fixed-wall step is modified.
- Rigid P2 now executes reduction -> moment finalize/3x3 solve -> rigid motion ->
  RK2 compact-state update in Warp stream order, without `.numpy()`,
  ParticleArray pull, or explicit synchronization.
- Compact rigid state/reduction are f64 even when particle fields are fp32;
  particle positions/velocities remain in the configured particle dtype.
- Both precisions match the NumPy implementation of PySPH's
  `RigidBodyMoments`, `RigidBodyMotion`, and `RK2StepRigidBody` at the stated
  tolerances. Existing P1 host-query behavior remains available.

## Tests / validation run

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py -k rigid \
    pysph/base/tests/test_warp_codegen.py::test_2d_path_generated_source_is_byte_identical_to_golden
..........                                                               [100%]
10 passed, 40 deselected, 2 warnings in 6.70s
```

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py
................................................                         [100%]
48 passed, 2 warnings in 1825.06s (0:30:25)
```

The full run was collected immediately before the final singular-geometry test
was added; that host-only guard is included in the subsequent 10-test focused
rerun above. A final warm-cache run of the exact final tree then passed all 49:

```text
$ python -m pytest -q --disable-warnings pysph/base/tests/test_warp_sph.py
.................................................                        [100%]
49 passed, 2 warnings in 12.84s
```

The 30-minute cold wall time is the known NVRTC compile of the
large fused Wendland 3D dam-break kernel; the process stayed healthy at full CPU
utilization and the remaining tests completed immediately afterward.

P2 repeated-step runtime checkpoint (RTX 4060, 315-particle asymmetric 3D box,
prescribed force + torque, 2,000 RK2 steps, `dt=1e-4`, one final pull):

```text
wall_s=0.5445955659997708; steps_per_s=3672.4500250537144
device_error=0; all_finite=true; vc_max_abs_error=1.552180384223334e-09
relative_pair_distance_drift=9.386771416218177e-07
```

This is a P2 dynamics/runtime checkpoint, not fluid-coupled validation.

```text
$ python -m py_compile pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py
<no output; exit 0>

$ git diff --check
<no output; exit 0>
```

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/update-decision-graph.py
Generated decisions/index.json and decisions/graph.md

$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

## Boundary amendment

- `implementation.md` boundary section updated: n/a — both host files match
  existing `pysph/base/warp_*.py` and `pysph/base/tests/test_warp_*.py` entries.
- Amendments log entry: n/a.

## Visual aid

| P2 stage | Previous P1/P2 proposal | Implemented production path |
| --- | --- | --- |
| Particle force/torque reduction | GPU f64 atomic sum | GPU f64 atomic sum |
| COM/inertia/torque finalize | Host NumPy | GPU f64 kernel |
| 3x3 angular solve | Host NumPy | GPU f64 symmetric solve |
| RK2 body-state update | Host | GPU f64 kernel |
| Particle rigid motion | GPU | GPU |
| Per-stage host boundary | reduction copy + solve | **none** |

The substantial fluid-only showcase images belong to—and are embedded in—the
[3D dam-break review](2026-06-19_warp-3d-dam-break-lobovsky.md#medium-resolution-developed-showcase-2026-06-20).
They validate the existing multi-array SPH path, not the still-pending P3
fluid/rigid coupling; keeping that distinction in the review prevents a pretty
image from becoming an accidental physics claim.

## Adversarial review

Reviewed five dimensions: 3x3/cofactor and torque algebra; RK2 stage ordering;
device residency/synchronization; dtype/device/lifecycle handling; and additive
compatibility/cache stability. Findings fixed before the final run:

- zero all compact outputs on a device-detected zero-mass reduction instead of
  leaving stale values;
- reject a state/ParticleArray device mismatch at setup;
- reject empty, zero-mass, and singular-inertia bodies once at setup while
  retaining the device error flag for runtime corruption;
- correct stale P1 comments that still described host finalize as production.

No unresolved correctness finding remains for the P2 boundary. Long-horizon
rotation drift and physical fluid coupling remain explicit P4/P3 risks below.

## Risks

- The explicit 3x3 solve requires non-singular body geometry. Setup validates
  every body once; the device kernel also emits an error flag for runtime
  corruption.
- RK2 position integration intentionally matches PySPH's particle-based rigid
  step rather than introducing a quaternion/orientation-matrix scheme. P4 must
  measure long-horizon rigidity drift in the assembled case.
- Four ordered GPU operations remain per stage (zero/reduce, finalize, particle
  motion, compact-state update). They have no host barrier and are compatible
  with later CUDA-graph capture; launch fusion is not part of P2.
- P3 must still compute physically correct Liu equal-and-opposite forces before
  this integrator is meaningful in the dam-break case.

## Unresolved questions

- Whether P4 needs quaternion/orientation-matrix integration depends on measured
  long-run geometry drift; do not change away from PySPH parity preemptively.
- Device error flags are intentionally read only at explicit validation/output
  boundaries. P3/P4 should define checkpoint failure reporting.

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > take this as @prabhu: LGTM for P2. approved for P3
- Timestamp: 2026-06-20T19:13:48 CEST
