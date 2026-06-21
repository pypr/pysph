---
type: experiment
id: 2026-06-15_warp-wcsph-euler-step
created: 2026-06-15T23:27:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: complete
last_checked: 2026-06-20T06:04:44 CEST
---

# Experiment: Warp WCSPH Euler and Leapfrog Step

## Headline

On PrediQT-02, the Warp WCSPH prototype now supports a one-step Euler chain and
a minimal KDK leapfrog step that keeps density, pressure, pressure-gradient
acceleration, velocity update, position update, and periodic position wrapping
on the device.

## Purpose

Wire the already-ported Warp kernels into minimal dynamics steps:

```text
rho <- summation density
p   <- isothermal EOS
a   <- inviscid pressure gradient
u   <- u + dt*a
x   <- x + dt*u
```

and:

```text
a_n     <- WCSPH acceleration(x_n)
u_half  <- u_n + 0.5*dt*a_n
x_np1   <- x_n + dt*u_half
wrap x_np1 into periodic bounds when requested
refresh NNPS from device x_np1
a_np1   <- WCSPH acceleration(x_np1)
u_np1   <- u_half + 0.5*dt*a_np1
```

This is not yet a full PySPH integrator or EllipticDrop solver path. It is a
small correctness milestone proving that equation outputs can be consumed by
later Warp kernels without host readback between stages.

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-wcsph-euler-step/run_correctness.sh
```

The wrapper uses the active venv if one is already active; otherwise it sources:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

The chained Warp step should produce the same density, pressure, acceleration,
velocity, and position values as CPU reference calculations for small fixtures.
The later kernels must not overwrite device-computed values with stale host
arrays.

## What To Expect

This experiment is a correctness case, not a benchmark. A successful run should
print a pytest summary like:

```text
29 passed, 2 warnings
```

The warnings are currently Python/Warp ctypes deprecation warnings and are not
part of the pass/fail signal.

The run includes these dynamics checks:

- `test_warp_euler_step_updates_velocity_and_position_on_device`: direct Euler
  update with known acceleration.
- `test_warp_wc_sph_euler_step_matches_cpu_expected_state`: full Euler WCSPH
  chain:

```text
rho <- summation density
p   <- isothermal EOS
a   <- inviscid pressure gradient
u   <- u + dt*a
x   <- x + dt*u
```

- `test_warp_leapfrog_kick_drift_and_wrap_update_device_state`: direct
  leapfrog half-kick, drift, and periodic position wrap.
- `test_warp_wc_sph_leapfrog_step_matches_cpu_expected_state`: full KDK
  leapfrog WCSPH chain.
- `test_uniform_grid_warp_nnps_can_rebuild_from_device_positions`: NNPS refresh
  after device-side position changes without pushing stale host positions.

Failure means either the device kernels disagree with the CPU reference values,
the device-updated state was overwritten by stale host arrays, or the NNPS
refresh did not see device-side coordinates.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
```

The key tests are:

- direct Euler velocity/position update;
- direct leapfrog kick, drift, and periodic wrap update;
- full `wc_sph_euler_step()` comparison against CPU-computed density, EOS,
  pressure-gradient acceleration, and final state.
- full `wc_sph_leapfrog_step()` comparison against CPU-computed KDK density,
  EOS, pressure-gradient acceleration, and final state.
- `UniformGridWarpNNPS.update(push=False)` rebuilds from device positions
  instead of stale host coordinates.

## Success Criteria

This experiment succeeds when:

- the Euler kernel updates velocity and position on the device;
- the leapfrog kick and drift kernels update velocity and position on the
  device;
- periodic wrapping keeps drifted coordinates inside supplied device-side
  bounds;
- `wc_sph_euler_step()` chains density, EOS, pgrad, and Euler update without
  intermediate host pull/push;
- `wc_sph_leapfrog_step()` recomputes acceleration after drift through an NNPS
  refresh that skips host pushes;
- focused tests compare actual state values, not just neighbor counts or
  checksums;
- all values are finite.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
26 passed, 2 warnings
```

Repeated-step checkpoint:

```text
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-wcsph-euler-step/run_correctness.sh
29 passed, 2 warnings in 3.05s
```

Hardware and runtime:

- host: PrediQT-02
- Python environment: PQT venv
- CPU: Intel(R) Core(TM) Ultra 7 155H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.79, 8188 MiB
- Warp: 1.14.0
- PySPH: editable install from this checkout

Interpretation:

- The new `push=False` mode is required for chained Warp calls; otherwise EOS,
  pgrad, Euler, or leapfrog refresh can clobber device-computed inputs with
  stale host arrays.
- `UniformGridWarpNNPS.update(push=False)` is now the first device-aware refresh
  path after positions move.
- Periodic support in this checkpoint is position wrapping. Periodic
  minimum-image neighbor distances and periodic cell lookup remain follow-up
  work.

## Conclusion

The first minimal Warp dynamics steps are correct for focused fixtures. Euler
and KDK leapfrog now have device-side correctness coverage, and NNPS can be
refreshed from device positions without forcing the ParticleArray host copy to
become the source of truth between steps.
