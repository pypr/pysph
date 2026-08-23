---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/plans/2026-06-16_warp-xsph-gaussian-adaptive-baseline.md
adrs: []
aspects_touched: [warp-backend, particle-memory, validation-benchmarks]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: lgtm
---

# Review - Warp XSPH, Gaussian, Adaptive DT, and Baseline Comparison

## Diff summary

- Added Gaussian kernel value/gradient support to Warp SPH equation kernels
  with `kernel='cubic'` as the compatibility default and `kernel='gaussian'`
  for elliptical-drop parity.
- Added `compute_xsph_correction()` and `leapfrog_drift_xsph()` so KDK
  leapfrog can drift with `u + ax` after the first half-kick.
- Added `compute_wcsph_adaptive_timestep()` with device-side `dt_cfl` and
  `dt_force` factors, device reductions, and a single scalar `dt` transfer.
- Updated `wc_sph_leapfrog_step()` with optional `xsph_eps`, `kernel`,
  `adaptive_dt`, `cfl`, `dt_min`, `dt_max`, and `return_dt`.
- Updated the Warp elliptical-drop runner to default to Gaussian, expose XSPH
  and adaptive-dt controls, record `dt_history`, and pull arrays only for final
  checkpoint/metrics output.
- Added a CPU/Warp comparison script that writes CPU/Warp `.npz` outputs and a
  side-by-side image.
- Updated experiment, aspect, daily, session, and current memory.

## Aspects touched and host files modified

- Aspects: `warp-backend`, `particle-memory`, `validation-benchmarks`.
- Host files:
  - `pysph/base/warp_sph.py`
  - `pysph/base/tests/test_warp_sph.py`
- Experiment/memory files:
  - `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`
  - `.ai/implementations/blast-from-the-past/plans/2026-06-16_warp-xsph-gaussian-adaptive-baseline.md`
  - `.ai/implementations/blast-from-the-past/current.md`
  - `.ai/implementations/blast-from-the-past/aspects/*/context.md`
  - `.ai/implementations/blast-from-the-past/updates/daily/2026-06-16.md`
  - `.ai/implementations/blast-from-the-past/updates/session-logs/2026-06-16_2227.md`

## Behavioral / numerical changes

- Gaussian uses PySPH's `Gaussian(dim)` convention:

```text
W(q, h) = (1/sqrt(pi))^dim * h^-dim * exp(-q^2), q < 3
dW/dq = -2*q*(1/sqrt(pi))^dim * h^-dim * exp(-q^2), q < 3
```

- XSPH follows PySPH's leapfrog correction form:

```text
ax_i += -eps * m_j * W_ij * 2/(rho_i + rho_j) * (u_i - u_j)
x_i += dt * (u_i + ax_i)
```

- Adaptive timestep factors follow the WCSPH momentum/integrator pattern:

```text
dt_cfl_i = max_j(abs(HIJ * VIJ.XIJ / RIJ^2) + c0)
dt_force_i = au_i^2 + av_i^2 + aw_i^2
dt = cfl * min(hmin/max(dt_cfl), sqrt(hmin/sqrt(max(dt_force))))
```

- The runner now exercises Gaussian + Tait + artificial viscosity + XSPH +
  adaptive dt in the smoke wrapper. Repeated stepping keeps particle state on
  device; adaptive stepping reads back only the reduced scalar timestep, and
  final full-array pulls are explicit output/plot checkpoints.
- The CPU comparison baseline is a PySPH-primitive baseline using
  `LinkedListNNPS`, `Gaussian`, and matching equation formulas. It is not yet
  the full PySPH `Application/Solver` production baseline.

## Tests / validation run

```text
$ python -m pytest pysph/base/tests/test_warp_sph.py -q
..................                                                       [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
18 passed, 2 warnings in 3.50s
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
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 22.79 ms  (cached)
Module pysph.base.warp_sph 1bd567e load on device 'cuda:0' took 9.35 ms  (cached)
{
  "adaptive_dt": true,
  "all_finite": true,
  "alpha": 0.1,
  "beta": 0.0,
  "c0": 20.0,
  "cfl": 0.25,
  "dt_last": 9.999999747378752e-06,
  "dt_max_used": 9.999999747378752e-06,
  "dt_min_used": 9.999999747378752e-06,
  "eos": "tait",
  "gamma": 7.0,
  "kernel": "gaussian",
  "particles": 204,
  "radius_scale": 3.0,
  "rho_max": 0.9998562335968018,
  "rho_min": 0.534595251083374,
  "time": 1.9999999494757503e-05,
  "xsph_eps": 0.5
}
```

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/compare_warp_pysph_elliptical_drop.py --nx 8 --steps 2 --dt 1.0e-5 --rho0 1.0 --c0 20.0 --p0 0.0 --alpha 0.1 --beta 0.0 --gamma 7.0 --xsph-eps 0.5 --adaptive-dt --cfl 0.25 --dt-min 1.0e-7 --dt-max 1.0e-5 --prefix comparison-smoke
{
  "cpu": {
    "all_finite": true,
    "dt_max_used": 1e-05,
    "dt_min_used": 1e-05,
    "kinetic_energy": 8078.179846214378,
    "particles": 204,
    "radius_max": 0.9978296023877065,
    "rho_max": 0.9998561964891306,
    "rho_min": 0.534595094929311,
    "time": 2e-05
  },
  "warp": {
    "all_finite": true,
    "kinetic_energy": 8078.179766857993,
    "particles": 204,
    "radius_max": 0.9978295868060059,
    "rho_max": 0.9998562335968018,
    "rho_min": 0.534595251083374,
    "time": 1.9999999494757503e-05
  }
}
```

```text
$ python -m py_compile pysph/base/warp_sph.py .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/compare_warp_pysph_elliptical_drop.py
```

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
....................................                                     [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
36 passed, 2 warnings in 3.45s
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
  remain the already-approved Python Warp prototype surface.

## Visual aid

- Side-by-side comparison image:
  `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/comparison-smoke.png`

| Case | Particles | rho_min | rho_max | radius_max | kinetic_energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU PySPH-primitive | 204 | 0.534595094929311 | 0.9998561964891306 | 0.9978296023877065 | 8078.179846214378 |
| Warp GPU | 204 | 0.534595251083374 | 0.9998562335968018 | 0.9978295868060059 | 8078.179766857993 |

## Risks

- The comparison baseline is not yet PySPH's full `Application/Solver` output.
  It uses PySPH primitives and matching formulas for a short smoke comparison.
- Adaptive dt currently transfers one scalar timestep per step to Python
  because Warp launches still need host scalar arguments.
- The Gaussian radius scale increases neighbor count relative to CubicSpline;
  larger production runs should record timing and memory pressure.
- Periodic support is still position wrapping only. True periodic-neighbor
  distances and cell lookup remain separate work.

## Unresolved questions

- What tolerance should gate the first full PySPH `Application/Solver` baseline
  comparison at `t=0.0008` and `t=0.0038`?
- Should the next production run use `c0=1400.0` and `nx=40` immediately, or
  ramp `c0`/`nx` separately to keep failure modes readable?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-16T23:04:00 CEST
