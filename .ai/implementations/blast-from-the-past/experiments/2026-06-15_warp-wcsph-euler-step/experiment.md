---
type: experiment
id: 2026-06-15_warp-wcsph-euler-step
created: 2026-06-15T23:27:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: active
last_checked: 2026-06-15T23:27:00 CET
---

# Experiment: Warp WCSPH Euler Step

## Headline

On PrediQT-02, a one-step Warp WCSPH prototype now keeps density, pressure,
pressure-gradient acceleration, velocity update, and position update on the
device and matches CPU reference values in focused tests.

## Purpose

Wire the already-ported Warp kernels into the first minimal dynamics step:

```text
rho <- summation density
p   <- isothermal EOS
a   <- inviscid pressure gradient
u   <- u + dt*a
x   <- x + dt*u
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

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
```

The key tests are:

- direct Euler velocity/position update;
- full `wc_sph_euler_step()` comparison against CPU-computed density, EOS,
  pressure-gradient acceleration, and final state.

## Success Criteria

This experiment succeeds when:

- the Euler kernel updates velocity and position on the device;
- `wc_sph_euler_step()` chains density, EOS, pgrad, and Euler update without
  intermediate host pull/push;
- focused tests compare actual state values, not just neighbor counts or
  checksums;
- all values are finite.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
26 passed, 2 warnings
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
  pgrad, or Euler can clobber device-computed inputs with stale host arrays.
- This proves a single device-side step. A repeated GPU simulation still needs
  a device-aware NNPS refresh after positions move.

## Conclusion

The first minimal Warp dynamics step is correct for focused fixtures. The next
engineering step is a repeated-step loop that can rebuild or update NNPS from
device positions without forcing the ParticleArray host copy to become the
source of truth between steps.
