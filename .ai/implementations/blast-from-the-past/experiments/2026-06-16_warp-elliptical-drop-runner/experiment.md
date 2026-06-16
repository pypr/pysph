---
type: experiment
id: 2026-06-16_warp-elliptical-drop-runner
created: 2026-06-16T12:30:00 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: active
last_checked: 2026-06-16T14:14:00 CEST
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

This is a GPU smoke run and output-generation checkpoint. It is not yet a
validated recreation of the published elliptical-drop benchmark.

## Missing Physics / Integration

The existing PySPH example uses Gaussian kernel, Tait EOS, artificial
viscosity, XSPH correction, adaptive timestep, and PySPH's full
Application/Solver stack. The current Warp runner uses CubicSpline summation
density, Tait EOS with per-particle `cs`, pressure-gradient acceleration plus
Monaghan-style artificial viscosity, and fixed-step KDK leapfrog.

## What To Expect

A successful smoke run should:

- create a non-empty circular particle patch;
- run a small number of Warp leapfrog steps;
- print JSON metrics;
- report `"all_finite": true`;
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
- final scalar bounds and kinetic energy are printed for inspection.

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

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/results-smoke.npz
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

This runner is the bridge from isolated Warp equation tests to a particle
dynamics workload. Once it runs reliably, the next work is to close the
remaining physics and integration gaps with XSPH or equivalent stabilization,
Gaussian kernel support, adaptive timestep logic, and eventually PySPH
Application/Solver integration.

The first ramp shows the current prototype can evolve finite states beyond the
tiny smoke case. Tait EOS and per-particle sound speed are now present in the
Warp path, and artificial viscosity uses pair-averaged `cs`. This is still
short of PySPH's full elliptical-drop formulation because XSPH, Gaussian kernel
support, adaptive timestep integration, and baseline output comparison remain
open.
