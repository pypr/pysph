---
type: experiment
id: 2026-06-16_warp-elliptical-drop-runner
created: 2026-06-16T12:30:00 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: active
last_checked: 2026-06-17T18:40:00 CEST
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
- XSPH correction in the device step path.
- Two density modes:
  - `summation`, retained as the original Warp KDK smoke/default path;
  - `continuity`, the PySPH Application parity path that computes `arho` and
    advances `rho` through WCSPH PEC-style stages on device.
- Device-computed adaptive timestep factors `dt_cfl` and `dt_force`.
- One scalar `dt` transfer from device to host per adaptive step; no full
  particle-array pulls during stepping.
- Final checkpoint/output pulls are explicit and used for metrics/plots.

The smoke comparison script uses PySPH CPU primitives (`LinkedListNNPS`,
`Gaussian`, and the same equation formulas) as a baseline. The resolved
comparison script uses PySPH's no-scheme `Application/Solver` path as the CPU
baseline.

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

The resolved comparison succeeds when:

- PySPH CPU Application and Warp GPU both reach the requested checkpoint times;
- metrics report `all_finite == true` at every checkpoint;
- runtime and average-step-time metrics are recorded;
- side-by-side images exist and include exact ellipse overlays.

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

Resolved PySPH Application vs Warp comparison, timestep-policy parity:

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-timestep-policy --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved --max-steps 10000000
```

Case:

```text
nx=100
particles=31417
c0=1400.0
Gaussian kernel
Tait EOS gamma=7.0
alpha=0.1 beta=0.0
XSPH eps=0.5
density_mode=continuity
warp_timestep_policy=pysph
adaptive timestep cfl=0.3 n_damp=50
checkpoints: 0.0008, 0.0038
```

Performance:

| Backend | Wall time (s) | Steps | Average step time (s) | dt_min | dt_mean | dt_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PySPH CPU Application | 233.97314716299297 | 1393 | 0.16796349401507032 | 2.2023818173548364e-06 | 2.7404840979225977e-06 | 2.780917055777183e-06 |
| Warp GPU | 23.629107111992198 | 1393 | 0.016962747388364823 | 2.7459356128852786e-09 | 2.727925340990668e-06 | 2.7813784981844947e-06 |

Overall wall-time speedup: `9.901903870258701x`.

Checkpoint metrics:

| Time | Backend | Major axis | Minor axis | Exact major | Exact minor | rho_min | rho_max | Kinetic energy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0008 | PySPH CPU | 1.0816640490130147 | 0.9220712094421513 | 1.0831034701687434 | 0.9232728243814081 | 0.9995725485493702 | 1.0055542309769985 | 7818.556269923258 |
| 0.0008 | Warp GPU | 1.0816634893417358 | 0.9220717549324036 | 1.0831034701687434 | 0.9232728243814081 | 0.9995719790458679 | 1.0055550336837769 | 7818.5528883068255 |
| 0.0038 | PySPH CPU | 1.4365264918961826 | 0.6964109650387659 | 1.4392190525454083 | 0.6948212631228 | 0.9975631121324761 | 1.002131063362072 | 7797.707446258537 |
| 0.0038 | Warp GPU | 1.4365261793136597 | 0.6964123249053955 | 1.4392190525454083 | 0.6948212631228 | 0.9975622892379761 | 1.0021320581436157 | 7797.707014434469 |

CPU-vs-Warp deltas:

| Time | Major axis delta | Minor axis delta | rho_min delta | rho_max delta | KE delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0008 | -5.596712788769054e-07 | 5.454902523016614e-07 | -5.695035022457162e-07 | 8.027067783800135e-07 | -0.0033816164323070552 |
| 0.0038 | -3.1258252297661215e-07 | 1.3598666296354978e-06 | -8.228944999855159e-07 | 9.947815438060559e-07 | -0.0004318240680731833 |

Resolved outputs:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-timestep-policy-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-timestep-policy-t0p0008000.png
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-timestep-policy-t0p0038000.png
```

Explicit Warp fp32 rerun:

The runner creates host arrays as `float64`, but `WarpDeviceHelper` casts float
properties to `compyle.config.get_config().use_double`; in this environment the
config is `False`, so the active Warp device arrays are `float32`. The
comparison loaders cast checkpoint arrays to `float64` for metrics/plotting,
which can hide the device precision in the saved summary.

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-fp32-warp-only --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/fp32 --skip-pysph-application --max-steps 10000000
```

Result:

```text
Warp fp32 wall time: 26.5732471299998 s
steps: 1393
average step time: 0.019076272167982626 s
dt_min: 2.7459356128852786e-09
dt_mean: 2.727925340990668e-06
dt_max: 2.7813784981844947e-06
all_finite: true at both checkpoints
```

Compared with the committed timestep-policy run's Warp timing
(`23.629107111992198` s), this explicit rerun is `1.1245980224328238x`
slower. Since both runs use the same fp32 device path, this is treated as
run-to-run/module-cache variance rather than a precision effect. Relative to
the committed PySPH CPU Application time (`233.97314716299297` s), the explicit
fp32 Warp-only rerun is `8.804838415808415x` faster.

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/fp32/resolved-nx100-fp32-warp-only-summary.json
```

Million-particle fixed-step CPU/GPU comparison:

```text
$ python pysph/examples/elliptical_drop_no_scheme.py --nx 565 --tf 0.000003732778967800475 --timestep 0.0000003732778967800475 --no-adaptive-timestep --n-damp 0 --pfreq 10 --fname million-pysph --directory .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/pysph --logfile '' --quiet
real 57.48

$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/warp/million-warp.npz
real 7.17
```

Case:

```text
nx=565
particles=1,002,885
steps=10
fixed dt=3.732778967800475e-07
tf=3.732778967800475e-06
c0=1400.0
Gaussian kernel
Tait EOS gamma=7.0
alpha=0.1 beta=0.0
XSPH eps=0.5
density_mode=continuity
```

Performance:

| Backend | Wall time (s) | Steps | Average step time (s) |
| --- | ---: | ---: | ---: |
| PySPH CPU Application | 57.48 | 10 | 5.747999999999999 |
| Warp GPU | 7.17 | 10 | 0.717 |

Overall wall-time speedup: `8.01673640167364x`.

Final-state CPU-vs-Warp deltas:

| Metric | Delta |
| --- | ---: |
| axis_x_abs | -1.2296967044633789e-07 |
| axis_y_abs | 4.773760275966765e-10 |
| rho_min | -9.119009991565008e-10 |
| rho_max | 1.4501548406542497e-08 |
| kinetic_energy | 8.523681572114583e-06 |

Both final checkpoints were finite. The comparison is intentionally fixed-step
to make the short ten-step CPU/GPU timing apples-to-apples without adaptive
damping or output-time policy effects.

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/million-cpu-gpu-10step-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/pysph/million-pysph_00010.hdf5
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step/warp/million-warp.npz
```

Million-particle fixed-step cache-reuse rerun:

After changing the Warp continuity-density PEC path to reuse one neighbor
cache per half-stage, the same 1,002,885-particle fixed-step benchmark was
rerun. The PySPH CPU Application baseline did not change, so the comparison
uses the previously recorded CPU wall time.

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-cache-reuse/warp/million-warp.npz
real 4.51
```

Performance:

| Backend / run | Wall time (s) | Steps | Average step time (s) | Speedup vs CPU |
| --- | ---: | ---: | ---: | ---: |
| PySPH CPU Application | 57.48 | 10 | 5.747999999999999 | 1.0 |
| Warp GPU before cache reuse | 7.17 | 10 | 0.717 | 8.01673640167364 |
| Warp GPU after cache reuse | 4.51 | 10 | 0.45099999999999996 | 12.7450110864745 |

Cache profile:

| Metric | Before | After |
| --- | ---: | ---: |
| neighbor-cache builds per step | 8 | 2 |
| average neighbors per cache | 44.873 | 44.873 |
| segmented step wall-time range | 0.36-0.46 s | 0.098335-0.138290 s |
| segmented cache-time range | 0.27-0.33 s | 0.036122-0.052126 s |

The new Warp result is `1.58980044345898x` faster than the previous Warp
million-particle run while keeping the same final metrics to the recorded
precision.

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-cache-reuse/million-cpu-gpu-10step-cache-reuse-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-cache-reuse/warp/million-warp.npz
```

Million-particle fused-equation (code-generation) rerun:

Per ADR-0003 the four continuity-stage neighbor-loop equations (pressure
gradient, artificial viscosity, continuity, XSPH) are now expressed as
composable `WarpEquation` blocks and fused by a dynamic code generator
(`pysph/base/warp_codegen.py`) into one generated kernel per PEC half-stage.
The same 1,002,885-particle fixed-step benchmark and physics were rerun. The
PySPH CPU Application baseline is unchanged.

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output .ai/.../million-cpu-gpu-10step-fused-eqns/warp/million-warp.npz
warm-cache wall samples: 6.04, 5.94, 4.89, 3.77, 4.05  (best 3.77; cold first-compile 6.15)
```

The 10-step headline wall is overhead/IO-bound (Warp init + 1M-particle mgrid
creation + 282MB npz write), so the wall is noisy and roughly flat versus the
4.51 s cache-reuse run (best case 3.77 s, `15.25x` versus CPU). The meaningful
gain is per-step, isolated by the segmented profile:

| Metric | After cache reuse | After equation fusion |
| --- | ---: | ---: |
| equation-kernel launches per step | 8 | 2 |
| equation-kernel time per step | ~0.064-0.088 s | 0.011-0.014 s |
| segmented step wall-time range | 0.098335-0.138290 s | 0.075883-0.097561 s |
| segmented cache-time range | 0.036122-0.052126 s | 0.033743-0.045973 s |

Equation-kernel time dropped roughly `5-6x` and steady-state step wall about
`25%`; register pressure from the single larger kernel did not reduce
throughput. The neighbor-cache build is now the dominant per-step cost
(~45-50% of step wall).

Numerical parity (fused vs the prior separate-kernel Warp run): positions,
densities, and pressures are identical to fp32 print precision; kinetic energy
differs by `-6.4e-09`. Versus the recorded CPU baseline the deltas match the
cache-reuse run (`x ~1e-7`, `rho ~1e-8`, `kinetic_energy 8.5e-06`). Both final
states finite.

Adaptive `nx=100` resolved guard (Warp-only) with the fused path reached
`t=0.0038` in `1393` steps (identical to the committed run), all finite, with
shape deltas `~4.8e-07`, density deltas `~1e-06`, and kinetic-energy delta
`1.45e-04` versus the committed Warp metrics. Because that run is
per-step-compute-bound (1393 steps), the fusion shows in wall time too: the
adaptive path now runs fused(1)+dt_factors(1)=2 traversals per stage instead of
5, and the Warp wall fell from the committed `23.63 s` to `10.83-14.33 s`
(cross-session, same step count).

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-fused-eqns/million-cpu-gpu-10step-fused-eqns-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-10step-fused-eqns/warp/million-warp.npz
```

Million-particle adaptive GPU probe:

```text
$ python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 565 --steps 1 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --adaptive-dt --cfl 0.3 --dt-min 1.0e-10 --dt-max 0.0000003732778967800475 --density-mode continuity --output .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/full-nx565-gpu/adaptive-one-step-probe.npz
real 4.87
```

The one-step adaptive probe was finite for 1,002,885 particles. A full
`nx=565`, `tf=0.0076` GPU-only run was not started after the user clarified
"no multi hour run".

Earlier continuity-density-only diagnostic run:

- Same CPU baseline and physics, but Warp used the old runner policy that capped
  `dt_max` to the initial timestep.
- Warp took 1804 steps versus PySPH's 1393.
- Warp `dt_max=2.1090202153573046e-06`, while PySPH grew to
  `dt_max=2.780917055777183e-06`.
- This explained the remaining step-count difference after density parity.

Earlier summation-density diagnostic run:

- Same CPU baseline and case settings.
- Warp used `density_mode=summation`, while PySPH CPU evolved density using
  `ContinuityEquation` through `WCSPHStep`.
- Warp took 4807 steps, with `dt_min=3.664420711313454e-10` and
  `dt_mean=7.905138339974642e-07`.
- At `t=0.0008`, Warp density ranged from `0.9505811929702759` to
  `1.0439165830612183` versus PySPH's `0.9995725485493702` to
  `1.0055542309769985`.
- This explained the earlier 2225/4807-step behavior: the comparison was not
  using the same density evolution.

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

### Grid-direct neighbor traversal (ADR-0004)

Per ADR-0004 the WCSPH continuity hot path stops materializing a flat CSR
neighbor list. The fused equation kernel (and the adaptive CFL dt-factors
kernel) walk the uniform-grid cell list directly with the support cutoff
inline, so `build_neighbor_cache_gpu` is never called on the continuity path;
the only spatial index that remains is the cheap cell-list build (`_build_grid`).

Segmented per-step profile (`nx=565`, 1,002,885 particles, 12 fixed steps,
2 warmup discarded, via `profile_grid_direct_neighbors.py`):

```text
$ PYTHONPATH=.ai/.../2026-06-16_warp-elliptical-drop-runner \
  python .ai/.../2026-06-16_warp-elliptical-drop-runner/profile_grid_direct_neighbors.py --nx 565 --steps 12 --warmup 2
flat_cache_builds_total: 0          (was 2 builds/step under the flat fused path)
grid_builds_per_step:    2.0        (cheap cell list; ~0.0004-0.0007 s each)
equation_launches/step:  2          (per-launch 0.023-0.025 s; absorbs the neighbor traversal)
step_wall_s steady:      0.059-0.064 (was 0.076-0.098 flat fused; ~25-35% lower)
kinetic_energy:          7854.1276  (delta vs flat fused -1.99e-06; all_finite True)
```

Neighbor work drops from three traversals per half-stage (count + fill to build
the flat list, then one equation read) to one (the equation cutoff walk). The
equation kernel's per-launch time rises because it now does the traversal that
the flat build used to do separately, but eliminating the two build traversals
plus the host readback and the large allocation nets a lower per-step wall.

Adaptive `nx=100` resolved guard (Warp-only, pysph timestep policy) with the
grid-direct path reached `t=0.0038` in `1393` steps -- identical to the
committed run -- all finite, with shape deltas `~2.4e-07/6.6e-07`, density
deltas `~9e-07`, and a kinetic-energy delta `1.19e-04` (relative `~1.5e-08`)
versus the committed Warp metrics. The Warp wall fell to `7.82 s` (committed
`23.63 s`; prior fused `10.83-14.33 s`), the cache-build removal compounding
over 1393 steps. The grid-direct dt-factors kernel keeps the `rij2>1e-12` inner
guard and adds the support cutoff to reproduce the flat neighbor set exactly,
which is what holds the substep count at `1393`.

Fresh same-session CPU-vs-Warp headlines (no reused numbers; CPU =
single-threaded PySPH Cython Application via `headline_million_100step.py`/the
resolved harness, Warp = grid-direct on RTX 4060 fp32):

```text
nx=100 resolved (real PySPH Application vs Warp, adaptive, identical 1393 steps):
  CPU 160.10 s (0.1149 s/step) | Warp 6.78 s (0.00487 s/step) | speedup 23.6x
1M particles, 100 fixed steps (n_damp=0, identical dt both sides):
  CPU 344.50 s (3.445 s/step) | Warp 8.37 s total (2.39 setup + 5.98 step; 0.0598 s/step)
  speedup 41.2x wall / 57.6x per-step | KE rel delta 1.6e-09 | all_finite True
```

The 1M / 100-step ratio is the more representative throughput number: at scale
the GPU parallelism dominates, and 100 steps dilute the one-time setup (Warp
setup is ~29% of its 8.37 s; pure stepping is 5.98 s). The earlier `~30x` figure
was an artifact of mixing a fresh Warp wall with a stale CPU baseline; these
same-session ratios supersede it.

Output:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-grid-direct/million-cpu-gpu-grid-direct-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-grid-direct/million-segmented-grid-direct-profile.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-grid-direct/adaptive-nx100-grid-direct-summary.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-grid-direct/fresh-headline-speedups.json
```

### Production GPU result -- RTX PRO 6000 Blackwell (sm_120)

Run by @kunalpuri-prediqt on a cloud box (NVIDIA RTX PRO 6000 Blackwell Server
Edition, 95 GiB, sm_120; Warp 1.14 / CUDA Toolkit 12.9 / driver 13.0), same
grid-direct fp32 code. Million-particle, 100 fixed steps, CPU single-threaded
PySPH Application vs Warp:

```text
CPU  357.43 s total (3.574 s/step), KE 7854.038961
Warp   1.32 s total (0.454 setup + 0.863 step; 0.008625 s/step), KE 7854.038955
speedup 271.5x wall / 414.4x per-step | KE rel delta 8.0e-10 | all_finite True
segmented: flat_cache_builds 0; steady step wall ~8.0 ms; equation ~1.6 ms/launch; grid build ~0.2 ms
```

Notes: per-step ~8.0 ms vs the RTX 4060 laptop's ~60 ms (~7.4x faster on this
GPU); the single-threaded CPU stays ~3.5 s/step, so the headline jumps from
`41x/57x` (4060) to `271x/414x` (Blackwell). The fused grid kernel cold-compiled
once (38.6 s) then loaded cached (~6 ms) on subsequent runs, confirming the
deterministic-name on-disk kernel cache on a fresh machine. Blackwell (sm_120)
ran with no code/build changes (BUILD.md section 10).

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/million-cpu-gpu-grid-direct/blackwell-rtxpro6000-million-100step-summary.json
```

### Cross-GPU throughput sweep (`gpu_perf_sweep.py`)

Per-GPU throughput vs particle count for the grid-direct continuity PEC step
(fp32), collected in `gpu-sweep/` (one JSON per GPU + a comparison `README.md`).
Four full sweeps recorded -- L40S (sm_89), RTX 5090 (sm_120), RTX PRO 6000
Blackwell (sm_120), B300 SXM6 (sm_103) -- plus the 4060 1M anchor.

**Headline speedup vs a single CPU core** at 1M (single-threaded PySPH Cython
~3.5 s/step; GPU per-step from the sweep): RTX 4060 ~`59x`, RTX 5090 ~`235x`,
L40S ~`321x`, RTX PRO 6000 ~`426x`, B300 ~`491x` (directly-measured CPU-vs-Warp
pairs corroborate: 4060 `57.6x`, RTX PRO 6000 `414x`). Speedup is ~flat with N
until the GPU knee, then falls.

Throughput at **1M particles** (particle-steps/s): RTX 4060 `1.68e7` < RTX 5090
`6.74e7` < L40S `9.19e7` < RTX PRO 6000 `1.22e8` < B300 `1.41e8`. Peak/sustained:
5090 ~`7.3e7`, L40S ~`9.9e7`, and **both Blackwell cards plateau at ~`1.43e8`**
(a likely bandwidth/occupancy ceiling -- the B300's edge is scale: it holds the
peak to 6M, reaches **78.5M particles**, and keeps a higher post-knee plateau).

Open finding: every card shows a **super-linear knee** whose location is
non-monotonic with VRAM (RTX PRO 6000 95 GiB knees at 6M; B300 268 GiB at 10M;
5090 32 GiB at 10M; L40S 44 GiB at 21M) -- so it is not a capacity ceiling but
an algorithmic/grid effect to profile (grid-build vs equation time). All sweep
points finite.

**Cost-of-compute lens ($)**: priced by NVIDIA Brev on-demand rates (2026-06-18:
B300 $9.49/hr, RTX PRO 6000 $2.63, L40S $1.06, RTX 5090 $0.78). `$ per billion
particle-steps` = `($/hr) / (throughput x 3600) x 1e9` at 1M: RTX 5090 **$0.0032**
(1.0x) ~= L40S **$0.0032** (1.0x) < RTX PRO 6000 **$0.0060** (1.9x) < B300
**$0.0187** (5.8x). The cheap cards do the same SPH work for **~6x less money**
than the B300 -- the datacenter cards buy **scale and latency** (78.5M particles
in one box; ~2x faster single step), not cost-per-work. Pick by constraint:
throughput-bound batch -> RTX 5090 / L40S; biggest problem or fastest turnaround
-> B300 / RTX PRO 6000.

**Two estimate-only lenses** (modeled from existing data; no measured power or
profiler run -- and none planned, so these are the final reported values, each
with its caveat):

- _Energy-to-solution ($ lens's physical cousin)_: `kJ per billion particle-steps`
  = `TDP_W / throughput x 1e6`, from datasheet board power (B300 1400 W, RTX PRO
  6000 600 W, 5090 575 W, L40S 350 W, 4060 115 W). At 1M: L40S **3.8** < RTX PRO
  6000 **4.9** < 4060 **6.9** < 5090 **8.5** < B300 **10.0** kJ. This **breaks the
  5090=L40S dollar tie**: same $0.0032/Gp-step, but the L40S uses **~2.2x fewer
  joules** per unit work (350 W @ 9.19e7 vs 575 W @ 6.74e7), so on a power-capped
  or owned fleet the L40S wins outright; the B300 is the least energy-efficient
  (~2.6x the L40S). Caveat: the step is not FLOP-bound so cards won't pull full
  TDP -- this is an upper-bound bracket, not a measured draw.
- _Roofline / memory-bandwidth utilization (analytic)_: `MBU = throughput x ~1.12
  KB/p-step / peak_BW` (B_eff = own state ~60 B + measured avg_neighbors 44.9 x
  ~24 B; peaks B300 8 TB/s HBM3E, GDDR7 cards 1.79, L40S 0.864). At 1M: L40S
  **~11.9%** > RTX PRO 6000 **~7.6%** > 4060 **~7.0%** > 5090 **~4.2%** > B300
  **~2.0%** of peak. **Partly corrects the earlier "bandwidth/occupancy-bound"
  guess**: even this generous upper-bound byte model leaves every card well under
  peak (B300 ~2% of 8 TB/s), so the ~1.43e8 Blackwell ceiling is **occupancy /
  launch / grid-build bound, NOT a memory wall** -- large untapped headroom,
  especially on the big-memory cards; the next perf win is occupancy/launch
  tuning, not faster memory. Caveat: model-based whole-step upper bound (ignores
  L2 reuse, includes grid-build), robust within ~2-3x; no profiler counter.

Figures (`gpu-sweep/plot_gpu_sweep.py`, pure matplotlib from the JSONs):

_Performance lens:_

![Speedup vs 1 CPU core @ 1M particles](gpu-sweep/speedup_vs_cpu_1M.png)

![Throughput vs particle count](gpu-sweep/throughput_vs_particles.png)

![Per-step wall time vs particle count](gpu-sweep/perstep_vs_particles.png)

_Cost-of-compute lens ($):_

![Cost per billion particle-steps @ 1M](gpu-sweep/cost_per_billion_1M.png)

![Cost per billion particle-steps vs particle count](gpu-sweep/cost_per_billion_vs_particles.png)

_Estimate-only lenses (energy + roofline):_

![Energy per billion particle-steps @ 1M (TDP estimate)](gpu-sweep/energy_per_gpstep_1M.png)

![Analytic memory-bandwidth utilization @ 1M](gpu-sweep/mbu_at_1M.png)

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/gpu-sweep/README.md
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/gpu-sweep/sweep-{l40s,rtx5090,rtxpro6000,b300}.json
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/gpu-sweep/{throughput_vs_particles,perstep_vs_particles,speedup_vs_cpu_1M,cost_per_billion_1M,cost_per_billion_vs_particles,energy_per_gpstep_1M,mbu_at_1M}.png
```

## Interpretation

The resolved `nx=100` timestep-policy run is now an apples-to-apples
Application-backed comparison for the current prototype. Warp keeps the repeated
state device-authoritative, evolves density through `arho`, follows PySPH's
early `n_damp` timestep growth policy, and matches the PySPH CPU Application's
Application-backed comparison for the current prototype. Warp keeps the repeated
state device-authoritative, evolves density through `arho`, follows PySPH's
early `n_damp` timestep growth policy, and matches the PySPH CPU Application's
step count, shape, density, and kinetic-energy metrics to small
floating-point-scale deltas at both checkpoint times.

The original summation-density resolved run is retained as diagnostic evidence,
not as a benchmark. It explains why Warp previously took thousands more
iterations: summation-density refreshes caused larger pressure/density
excursions and collapsed the force timestep. Continuity-density staging fixed
the physics mismatch, and PySPH-like timestep policy fixed the remaining step
count mismatch.

The final resolved run gives exact step-count parity (`1393` CPU and `1393`
Warp steps) and `9.901903870258701x` wall-time speedup on this machine. The
tiny Warp `dt_min` is from an output-time landing step; it no longer affects
subsequent timestep growth.
