---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/plans/2026-06-16_warp-artificial-viscosity-momentum-term.md
adrs: []
aspects_touched: [warp-backend, gpu-nnps, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: lgtm
---

# Review - Warp Artificial Viscosity Momentum Term

## Diff summary

- Added float32/float64 Warp kernels for Monaghan-style artificial viscosity.
- Added `compute_artificial_viscosity()` as an additive acceleration pass over
  existing `au`, `av`, and `aw`.
- Wired optional `alpha`/`beta` through `wc_sph_leapfrog_step()` and
  `wc_sph_euler_step()` while preserving inviscid defaults.
- Added a focused CPU-reference test for artificial viscosity, including
  nonzero initial acceleration to prove additive behavior.
- Exposed `--alpha` and `--beta` in the Warp elliptical-drop runner and updated
  smoke/ramp experiment evidence.
- Updated current, aspect, daily, session, and experiment memory.

## Aspects touched and host files modified

- Aspects: `warp-backend`, `gpu-nnps`, `validation-benchmarks`,
  `host-integration`.
- Host files:
  - `pysph/base/warp_sph.py`
  - `pysph/base/tests/test_warp_sph.py`

## Behavioral / numerical changes

- New optional artificial-viscosity contribution:

```text
Pi_ij = (-alpha*c0*mu_ij + beta*mu_ij^2) * 2/(rho_i + rho_j)
mu_ij = HIJ * (v_ij dot x_ij) / (r_ij^2 + 0.01*HIJ^2)
```

- The term only contributes for approaching pairs where `v_ij dot x_ij < 0`.
- The viscosity pass adds onto existing acceleration arrays. The pressure
  gradient path still sets the inviscid acceleration first.
- Defaults remain inviscid in low-level step helpers: `alpha=0.0`,
  `beta=0.0`.
- The elliptical-drop runner now defaults to and records `alpha=0.1`,
  `beta=0.0`.

## Tests / validation run

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
..............................                                           [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 2 warnings in 4.39s
```

```text
$ bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "CPU"
     "cuda:0"   : "NVIDIA GeForce RTX 4060 Laptop GPU" (8 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/kunalp/.cache/warp/1.14.0
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 15.35 ms  (cached)
Module pysph.base.warp_sph 128be63 load on device 'cuda:0' took 4.16 ms  (cached)
{
  "all_finite": true,
  "alpha": 0.1,
  "beta": 0.0,
  "c0": 20.0,
  "dt": 1e-05,
  "kinetic_energy": 8078.167363381624,
  "nx": 8,
  "p_max": -0.01952648162841797,
  "p_min": -166.61477661132812,
  "particles": 204,
  "radius_max": 0.9978744032287784,
  "rho_max": 0.999951183795929,
  "rho_min": 0.5834630727767944,
  "steps": 2,
  "time": 2e-05,
  "x_max": 0.9480998516082764,
  "x_min": -0.9231499433517456,
  "y_max": 0.9518998861312866,
  "y_min": -0.9268498420715332
}
```

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 16 --steps 5 --dt 1.0e-5 --rho0 1.0 --c0 20.0 --p0 0.0 --alpha 0.1 --beta 0.0 --output .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-avisc-nx16-steps5.npz
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "CPU"
     "cuda:0"   : "NVIDIA GeForce RTX 4060 Laptop GPU" (8 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/kunalp/.cache/warp/1.14.0
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 12.47 ms  (cached)
Module pysph.base.warp_sph 128be63 load on device 'cuda:0' took 2.96 ms  (cached)
{
  "all_finite": true,
  "alpha": 0.1,
  "beta": 0.0,
  "c0": 20.0,
  "dt": 1e-05,
  "kinetic_energy": 7868.737673401772,
  "nx": 16,
  "p_max": -0.009822845458984375,
  "p_min": -146.79534912109375,
  "particles": 805,
  "radius_max": 1.0018194069173603,
  "rho_max": 0.9999754428863525,
  "rho_min": 0.6330116391181946,
  "steps": 5,
  "time": 5e-05,
  "x_max": 0.9452486038208008,
  "x_min": -0.9825611710548401,
  "y_max": 0.9547483921051025,
  "y_min": -0.992435872554779
}
```

```text
$ git diff --check -- pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past
```

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: n-a
- Amendments log entry: n-a
- Note: the plan is marked `within_boundary: false` because the memory
  validator treats `pysph/base/warp_*.py` literally, not as a glob. These files
  are still the already-approved Python Warp prototype surface.

## Visual aid

| Stage | Acceleration ownership | Host/device transfer posture |
| --- | --- | --- |
| `compute_pressure_gradient` | Sets inviscid `au/av/aw` | Pushes only when requested |
| `compute_artificial_viscosity` | Adds viscosity to `au/av/aw` | Uses device arrays with `push=False` inside integrators |
| `wc_sph_leapfrog_step` | Recomputes forces before and after drift | Keeps repeated steps device-authoritative |

## Risks

- Constant `c0` is an approximation until Tait EOS and per-particle sound speed
  are added.
- CubicSpline remains the active Warp kernel; the PySPH elliptical-drop example
  uses Gaussian.
- Periodic support is still coordinate wrapping only, not minimum-image
  neighbor interaction.

## Unresolved questions

- What correctness threshold should promote the Warp elliptical-drop runner
  from smoke workload to benchmark?
- Should the next physics slice add Tait EOS/per-particle `cs` first, or XSPH
  stabilization first?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-16T13:52:32 CEST
