---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: plans/2026-06-16_warp-elliptical-drop-application-runner.md
adrs: []
aspects_touched: [validation-benchmarks, gpu-nnps, warp-backend, particle-memory, host-integration]
host_files: []
status: approved
---

# Review - Warp elliptical drop runner smoke ramp

## Diff summary

- Adds an application-style Warp elliptical-drop runner under the active
  experiment tree.
- Adds a `run_correctness.sh` smoke wrapper.
- Adds an experiment document with expectations, success criteria, raw smoke
  output, and ramp metrics.
- Updates current/aspect/daily/session memory with the finite smoke/ramp
  results.
- Leaves generated `.npz` result files as local experiment artifacts, not
  intended for commit unless explicitly requested.

## Aspects touched and host files modified

- `validation-benchmarks`: new runnable experiment and ramp table.
- `gpu-nnps`: exercises `UniformGridWarpNNPS` in a dynamics workload.
- `warp-backend`: exercises Warp kernels through an application-style runner.
- `particle-memory`: pulls final arrays once at the end of each run.
- `host-integration`: no PySPH `Application`/`Solver` integration yet.

Host files modified: none. This slice only changes implementation memory and
experiment files.

## Behavioral / numerical changes

- No host package behavior changes.
- New runner creates the standard elliptical-drop circular particle patch and
  initial velocity field, then advances it with:

```text
UniformGridWarpNNPS
wc_sph_leapfrog_step
```

- This is a GPU state-evolution smoke/ramp workload, not a validated published
  elliptical-drop benchmark. Current Warp physics still lacks Tait EOS,
  artificial viscosity, XSPH, Gaussian kernel support, and adaptive timestep.

## Tests / validation run

```text
$ bash .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/run_correctness.sh
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "CPU"
     "cuda:0"   : "NVIDIA GeForce RTX 4060 Laptop GPU" (8 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/kunalp/.cache/warp/1.14.0
Module pysph.base.warp_nnps b046253 load on device 'cuda:0' took 25.10 ms  (cached)
Module pysph.base.warp_sph e548a6b load on device 'cuda:0' took 6.87 ms  (cached)
{
  "all_finite": true,
  "dt": 1e-05,
  "kinetic_energy": 8078.22338525834,
  "nx": 8,
  "p_max": -0.01952648162841797,
  "p_min": -166.61477661132812,
  "particles": 204,
  "radius_max": 0.9978743942869612,
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

Ramp runs recorded in the experiment doc:

```text
nx=12, particles=455, steps=5,  time=5e-05,  all_finite=true
nx=16, particles=805, steps=5,  time=5e-05,  all_finite=true
nx=16, particles=805, steps=20, time=0.0001, all_finite=true
nx=24, particles=1808, steps=10, time=5e-05,  all_finite=true
```

```text
$ git diff --check -- .ai/implementations/blast-from-the-past
<no output>
```

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: n-a
- Amendments log entry: n-a

## Visual aid

| Case | Particles | Steps | Time | Result |
| --- | ---: | ---: | ---: | --- |
| smoke nx=8 | 204 | 2 | 2e-05 | finite |
| ramp nx=12 | 455 | 5 | 5e-05 | finite |
| ramp nx=16 | 805 | 5 | 5e-05 | finite |
| ramp nx=16 | 805 | 20 | 0.0001 | finite |
| ramp nx=24 | 1808 | 10 | 5e-05 | finite |

## Risks

- The runner is not the full PySPH elliptical-drop application and should not
  be compared against the analytical benchmark yet.
- Current pressure model can produce negative pressure with the prototype
  isothermal EOS and incomplete WCSPH formulation.
- Generated `.npz` files are local artifacts and are not included in the
  intended commit.

## Unresolved questions

- Should generated `.npz` experiment artifacts be committed or kept local?
- What ramp target should gate moving to artificial viscosity: larger `nx`,
  longer time, or comparison to a PySPH baseline?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-16T13:25:52 CEST
