---
type: experiment
id: 2026-06-16_warp-elliptical-drop-runner
created: 2026-06-16T12:30:00 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: active
last_checked: 2026-06-16T22:56:00 CEST
---

# Experiment: Warp Elliptical-Drop Runner

## Headline

First application-style runner for the elliptical-drop initial condition using
the current Warp NNPS and WCSPH leapfrog prototype.

## Purpose

Create the same circular patch and initial velocity field used by PySPH's
elliptical-drop example, then advance it with:

```text
UniformGridWarpNNPS
wc_sph_leapfrog_step
```

This is now a GPU smoke/comparison run. It exercises the same core terms used
by the PySPH no-scheme elliptical-drop example: Gaussian kernel, Tait EOS,
artificial viscosity, XSPH correction, and adaptive timestep control.

## Active Physics / Integration

- Gaussian kernel with `radius_scale=3.0` by default.
- Tait EOS with per-particle sound speed `cs`.
- Pressure-gradient acceleration plus additive Monaghan artificial viscosity.
- XSPH correction in the leapfrog drift path.
- Device-computed adaptive timestep factors `dt_cfl` and `dt_force`.
- One scalar `dt` transfer from device to host per adaptive step; no full
  particle-array pulls during stepping.
- Final checkpoint/output pulls are explicit and used for metrics/plots.

The remaining gap is full PySPH `Application/Solver` parity for a production
elliptical-drop run. The current comparison script uses PySPH CPU primitives
(`LinkedListNNPS`, `Gaussian`, and the same equation formulas) as a baseline.

## What To Expect

A successful smoke run should:

- create a non-empty circular particle patch;
- run a small number of Warp leapfrog steps;
- print JSON metrics;
- report `"all_finite": true`;
- record `"kernel": "gaussian"`;
- record `"xsph_eps": 0.5`;
- record `"adaptive_dt": true`;
- write `results-smoke.npz`.

The default smoke settings are intentionally conservative:

```text
nx=8
steps=2
dt=1.0e-5
c0=20.0
alpha=0.1
beta=0.0
eos=tait
gamma=7.0
kernel=gaussian
xsph_eps=0.5
adaptive_dt=true
cfl=0.25
```

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh
```

## Success Criteria

This experiment succeeds when:

- the wrapper exits with status 0;
- output file `results-smoke.npz` exists and is non-empty;
- metrics report `particles > 0`;
- metrics report `all_finite == true`;
- metrics record `alpha == 0.1` and `beta == 0.0`;
- metrics record `eos == "tait"` and `gamma == 7.0`;
- metrics record `kernel == "gaussian"`, `radius_scale == 3.0`,
  `xsph_eps == 0.5`, and `adaptive_dt == true`;
- metrics record finite `dt_min_used`, `dt_max_used`, and `dt_last`;
- final scalar bounds and kinetic energy are printed for inspection.

The comparison script succeeds when:

- CPU and Warp `.npz` outputs exist;
- CPU and Warp metrics report `all_finite == true`;
- side-by-side image `comparison-smoke.png` exists and is non-empty.

## Results

Smoke run:

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
  "cs_max": 19.991374969482422,
  "cs_min": 3.055661916732788,
  "dt": 1e-05,
  "dt_last": 9.999999747378752e-06,
  "dt_max": 1e-05,
  "dt_max_used": 9.999999747378752e-06,
  "dt_min": 1e-07,
  "dt_min_used": 9.999999747378752e-06,
  "eos": "tait",
  "gamma": 7.0,
  "kernel": "gaussian",
  "kinetic_energy": 8078.179766857993,
  "nx": 8,
  "p_max": -0.05748271942138672,
  "p_min": -56.429779052734375,
  "particles": 204,
  "radius_max": 0.9978295868060059,
  "radius_scale": 3.0,
  "rho_max": 0.9998562335968018,
  "rho_min": 0.534595251083374,
  "steps": 2,
  "time": 1.9999999494757503e-05,
  "x_max": 0.9481551647186279,
  "x_min": -0.9232051968574524,
  "xsph_eps": 0.5,
  "y_max": 0.951850950717926,
  "y_min": -0.9268011450767517
}
```

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-smoke.npz
```

CPU/Warp side-by-side comparison smoke:

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
    "kernel": "gaussian",
    "kinetic_energy": 8078.179766857993,
    "particles": 204,
    "radius_max": 0.9978295868060059,
    "rho_max": 0.9998562335968018,
    "rho_min": 0.534595251083374,
    "time": 1.9999999494757503e-05,
    "xsph_eps": 0.5
  }
}
```

Comparison outputs:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/comparison-smoke-cpu.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/comparison-smoke-warp.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/comparison-smoke.png
```

Ramp runs:

| Case | Particles | Steps | dt | Time | all_finite | rho_min | rho_max | radius_max | kinetic_energy |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| nx=12 | 455 | 5 | 1e-05 | 5e-05 | true | 0.6329907178878784 | 0.9999749660491943 | 1.0018477038507216 | 7943.001147793795 |
| nx=16 | 805 | 5 | 1e-05 | 5e-05 | true | 0.6330121159553528 | 0.9999753832817078 | 1.0018193926728525 | 7868.808903639647 |
| nx=16 | 805 | 20 | 5e-06 | 0.0001 | true | 0.6331153512001038 | 1.0000600814819336 | 1.0066060209042353 | 7868.821050761739 |
| nx=24 | 1808 | 10 | 5e-06 | 5e-05 | true | 0.633074939250946 | 0.9999793767929077 | 1.0033569350841507 | 7840.533230601928 |

Artificial-viscosity ramp check:

| Case | alpha | beta | Particles | Steps | dt | Time | all_finite | rho_min | rho_max | radius_max | kinetic_energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| nx=16 | 0.1 | 0.0 | 805 | 5 | 1e-05 | 5e-05 | true | 0.6330116391181946 | 0.9999754428863525 | 1.0018194069173603 | 7868.737673401772 |

Tait EOS + per-particle sound-speed ramp check:

| Case | eos | gamma | alpha | beta | Particles | Steps | dt | Time | all_finite | rho_min | rho_max | cs_min | cs_max | radius_max | kinetic_energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nx=16 | tait | 7.0 | 0.1 | 0.0 | 805 | 5 | 1e-05 | 5e-05 | true | 0.6329819560050964 | 0.9999754428863525 | 5.072288990020752 | 19.99852752685547 | 1.0018218256790075 | 7868.739071212255 |

Ramp output files:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-ramp-nx12-steps5.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-ramp-nx16-steps5.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-ramp-nx16-steps20.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-ramp-nx24-steps10.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-avisc-nx16-steps5.npz
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-tait-nx16-steps5.npz
```

## Interpretation

This runner has crossed from isolated equation tests to a near-formulation
match for the no-scheme elliptical-drop physics. The small comparison smoke is
intentionally short, but it shows the Warp GPU and CPU PySPH-primitive paths
agree closely for density bounds, radius, kinetic energy, and visual layout.

The next escalation should be a longer production-oriented run using the full
PySPH `Application/Solver` output as the baseline at `t=0.0008` and
`t=0.0038`, then compare major/minor-axis metrics and images.
