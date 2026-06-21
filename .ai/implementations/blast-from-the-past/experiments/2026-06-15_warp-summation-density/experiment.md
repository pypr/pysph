---
type: experiment
id: 2026-06-15_warp-summation-density
created: 2026-06-15T12:05:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: complete
last_checked: 2026-06-20T06:04:44 CEST
---

# Experiment: Warp Summation Density

## Headline

At 10,000,000 particles on PrediQT-02, `warp_grid_density` computes SPH
summation density in `3.330 s` versus PySPH CPU/Cython `230.051 s`, a
`69.084x` speedup, with matching checksum on Intel(R) Core(TM) Ultra 7 155H CPU
and NVIDIA GeForce RTX 4060 Laptop GPU.

## Purpose

Port a first real SPH equation kernel to Warp and measure operation speedup
against the CPU path.

The target equation is PySPH's standard
`pysph.sph.basic_equations.SummationDensity`:

```text
rho_i = sum_j m_j * W_ij
```

The Warp implementation uses the device-resident `UniformGridWarpNNPS` neighbor
cache and the same generated-equation convention as PySPH:

```text
HIJ = 0.5 * (d_h[d_idx] + s_h[s_idx])
WIJ = CubicSpline(XIJ, RIJ, HIJ)
```

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-summation-density/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-summation-density/run_benchmark.sh --sizes 1000000 2000000 5000000 10000000 --repeats 1
```

The wrappers use the active venv if one is already active; otherwise they
source:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

The Warp path should match CPU density checksums and significantly outperform
PySPH CPU/Cython for million-particle uniform random 2D cases, because the
neighbor cache and density loop stay on the GPU.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
```

`run_benchmark.sh` runs `benchmark_summation_density.py`, which records:

- CPU model;
- GPU model, driver, and memory;
- Warp version;
- backend;
- particle count;
- repeat count;
- p50 operation time;
- density checksum;
- speedup relative to `cpu_cython`.

CPU baseline:

- PySPH `SPHEvaluator`;
- `SummationDensity(dest='fluid', sources=['fluid'])`;
- `CubicSpline(dim=2)`;
- `LinkedListNNPS`;
- Cython backend.

Warp baseline:

- `UniformGridWarpNNPS`;
- `compute_summation_density()`;
- device-resident neighbor cache plus Warp CubicSpline density kernel.

## Success Criteria

This experiment succeeds when:

- focused correctness tests compare Warp density values against a CPU
  `CubicSpline` reference;
- CPU and Warp benchmark checksums match to reported precision;
- the benchmark completes the requested 1M-to-10M sweep;
- timing output reports CPU-relative speedup;
- CPU and GPU hardware are listed.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
19 passed
```

Hardware and runtime:

- host: PrediQT-02
- Python environment: PQT venv
- Python executable: `/home/kunalp/.pqt_venv_e0b41259/bin/python`
- CPU: Intel(R) Core(TM) Ultra 7 155H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.79, 8188 MiB
- Warp: 1.14.0
- PySPH: editable install from this checkout

1,000,000-particle benchmark:

```text
backend particles repeats p50_ms checksum speedup_vs_cpu status
cpu_cython          1000000       1 16714.931   1.230047e+06          1.000 ok
warp_grid_density   1000000       1   109.601   1.230047e+06        152.508 ok
```

2M-to-10M sweep:

```text
backend particles repeats p50_ms checksum speedup_vs_cpu status
cpu_cython          2000000       1  40101.093   2.463004e+06          1.000 ok
warp_grid_density   2000000       1    176.226   2.463004e+06        227.555 ok
cpu_cython          5000000       1 104574.437   6.159220e+06          1.000 ok
warp_grid_density   5000000       1   1609.723   6.159220e+06         64.964 ok
cpu_cython         10000000       1 230050.832   1.231664e+07          1.000 ok
warp_grid_density  10000000       1   3330.035   1.231664e+07         69.084 ok
```

Interpretation:

- Warp density checksums match CPU/Cython to the reported precision for all
  measured sizes.
- The 1M and 2M cases show especially high speedup because the Warp path fits
  comfortably and the operation is dominated by parallel neighbor work.
- 5M and 10M still show large speedups, but the speedup is lower than 2M. This
  likely reflects the larger flat neighbor cache, device memory pressure, and
  cache construction cost on the 8 GiB laptop GPU.
- The 10M result is the current best proof that the pipeline is now doing real
  SPH work on the GPU: Warp NNPS cache plus Warp CubicSpline summation density.

## Conclusion

The first real SPH equation kernel is now running on the Warp path and shows
large CPU-relative speedups through 10,000,000 particles. The next target is to
generalize this one-equation path into a reusable Warp equation execution
contract and then add the next equations needed for a minimal solver step.
