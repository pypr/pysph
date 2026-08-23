---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/plans/2026-06-16_warp-tait-eos-and-sound-speed.md
adrs: []
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: lgtm
---

# Review - Warp Tait EOS and Sound Speed

## Diff summary

- Added Warp float32/float64 Tait EOS kernels and `compute_tait_eos()`.
- Added per-particle sound speed `cs` handling for WCSPH force evaluation.
- Updated artificial viscosity to use `cij = 0.5*(d_cs + s_cs)` with a
  constant-`c0` fallback when callers have not created `cs`.
- Added `eos='tait'` and `gamma` options to the WCSPH Euler/leapfrog helpers
  while preserving `eos='isothermal'` as the compatibility default.
- Switched the Warp elliptical-drop runner default to Tait EOS with
  `gamma=7.0` and included `cs` metrics/output.
- Updated experiment and memory artifacts with Tait smoke/ramp results.

## Aspects touched and host files modified

- Aspects: `warp-backend`, `gpu-nnps`, `particle-memory`,
  `validation-benchmarks`, `host-integration`.
- Host files:
  - `pysph/base/warp_sph.py`
  - `pysph/base/tests/test_warp_sph.py`

## Behavioral / numerical changes

- Tait EOS now computes:

```text
ratio = rho / rho0
p = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1)
cs = c0 * ratio**(0.5*(gamma - 1))
```

- Artificial viscosity now uses pair-averaged sound speed:

```text
cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])
Pi_ij = (-alpha*cij*mu_ij + beta*mu_ij^2) * 2/(rho_i + rho_j)
```

- The low-level WCSPH helpers still default to isothermal EOS for existing
  tests and callers.
- The elliptical-drop runner now defaults to `eos=tait`, `gamma=7.0`.
- This does not make elliptical drop production-ready. XSPH, Gaussian kernel
  support or a kernel decision, adaptive timestep/CFL tracking, and PySPH
  baseline comparison remain open.

## Tests / validation run

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
................................                                         [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
32 passed, 2 warnings in 4.14s
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
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 16.14 ms  (cached)
Module pysph.base.warp_sph 37ca4ca load on device 'cuda:0' took 4.78 ms  (cached)
{
  "all_finite": true,
  "alpha": 0.1,
  "beta": 0.0,
  "c0": 20.0,
  "cs_max": 19.997066497802734,
  "cs_min": 3.9725253582000732,
  "dt": 1e-05,
  "eos": "tait",
  "gamma": 7.0,
  "kinetic_energy": 8078.17389338273,
  "nx": 8,
  "p_max": -0.01954691670835018,
  "p_min": -55.82748794555664,
  "particles": 204,
  "radius_max": 0.9978746006297383,
  "rho_max": 0.9999511241912842,
  "rho_min": 0.5834615230560303,
  "steps": 2,
  "time": 2e-05,
  "x_max": 0.9480999112129211,
  "x_min": -0.9231499433517456,
  "y_max": 0.9518999457359314,
  "y_min": -0.9268499612808228
}
```

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 16 --steps 5 --dt 1.0e-5 --rho0 1.0 --c0 20.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --output .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-tait-nx16-steps5.npz
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "CPU"
     "cuda:0"   : "NVIDIA GeForce RTX 4060 Laptop GPU" (8 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/kunalp/.cache/warp/1.14.0
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 15.20 ms  (cached)
Module pysph.base.warp_sph 37ca4ca load on device 'cuda:0' took 3.32 ms  (cached)
{
  "all_finite": true,
  "alpha": 0.1,
  "beta": 0.0,
  "c0": 20.0,
  "cs_max": 19.99852752685547,
  "cs_min": 5.072288990020752,
  "dt": 1e-05,
  "eos": "tait",
  "gamma": 7.0,
  "kinetic_energy": 7868.739071212255,
  "nx": 16,
  "p_max": -0.009822845458984375,
  "p_min": -54.81636428833008,
  "particles": 805,
  "radius_max": 1.0018218256790075,
  "rho_max": 0.9999754428863525,
  "rho_min": 0.6329819560050964,
  "steps": 5,
  "time": 5e-05,
  "x_max": 0.9452491998672485,
  "x_min": -0.9825617671012878,
  "y_max": 0.9547492861747742,
  "y_min": -0.9924367666244507
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

| Path | Pressure | Sound speed | Viscosity sound speed |
| --- | --- | --- | --- |
| `eos='isothermal'` | `p0 + c0^2*(rho-rho0)` | `c0` fallback | constant `c0` unless caller supplies `cs` |
| `eos='tait'` | Tait pressure | per-particle `cs` | `0.5*(d_cs+s_cs)` |
| elliptical-drop runner | Tait by default | written to output | pair-averaged `cs` |

## Risks

- Tait EOS fractional powers require positive density ratios. Current smoke
  and ramp stay finite, but larger/longer runs still need guards or acceptance
  thresholds.
- The runner's default metrics changed because it now uses Tait EOS instead of
  isothermal EOS.
- Production elliptical drop still needs XSPH, Gaussian kernel support or an
  accepted kernel decision, adaptive timestep/CFL logic, and baseline
  comparison.

## Unresolved questions

- Should the next physics slice add XSPH first or Gaussian kernel support first?
- What exact PySPH output times/tolerances should gate the first
  elliptical-drop benchmark comparison?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-16T17:25:35 CEST
