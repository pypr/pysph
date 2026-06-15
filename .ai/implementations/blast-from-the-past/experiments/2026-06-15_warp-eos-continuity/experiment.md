---
type: experiment
id: 2026-06-15_warp-eos-continuity
created: 2026-06-15T13:05:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: active
last_checked: 2026-06-15T13:25:00 CET
---

# Experiment: Warp EOS And Continuity

## Headline

At 5,000,000 particles on PrediQT-02, `warp_grid_eos_cont` computes
IsothermalEOS plus ContinuityEquation in `1.799 s` versus PySPH CPU/Cython
`130.549 s`, a `72.583x` speedup, on Intel(R) Core(TM) Ultra 7 155H CPU and
NVIDIA GeForce RTX 4060 Laptop GPU.

## Purpose

Add the next two simple SPH equation kernels after summation density:

- `IsothermalEOS`: `p = p0 + c0^2*(rho - rho0)`;
- `ContinuityEquation`: `arho_i = sum_j m_j * VIJ . DWIJ`.

This experiment measures the paired operation against PySPH's CPU/Cython
execution path and intentionally caps the sweep at 5M particles.

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-eos-continuity/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-eos-continuity/run_benchmark.sh --sizes 1000000 2000000 5000000 --repeats 1
```

The wrappers use the active venv if one is already active; otherwise they
source:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

EOS should be memory-bandwidth friendly and continuity should behave similarly
to the summation-density benchmark while adding velocity and gradient work. The
Warp pair should match CPU/Cython checksums to useful aggregate precision and
provide large operation speedups at million-particle scales.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
```

`run_benchmark.sh` runs `benchmark_eos_continuity.py`, which records:

- CPU model;
- GPU model, driver, and memory;
- Warp version;
- backend;
- particle count;
- repeat count;
- p50 operation time;
- pressure checksum;
- density-rate checksum;
- speedup relative to `cpu_cython`.

CPU baseline:

- PySPH `SPHEvaluator`;
- Cython backend;
- `IsothermalEOS`;
- `ContinuityEquation`;
- `CubicSpline(dim=2)`;
- `LinkedListNNPS`.

Warp baseline:

- `compute_isothermal_eos()`;
- `compute_continuity()`;
- `UniformGridWarpNNPS` device neighbor cache;
- Warp CubicSpline gradient kernel.

## Success Criteria

This experiment succeeds when:

- focused tests compare Warp EOS and continuity values against CPU references;
- CPU and Warp benchmark checksums are close enough to catch major correctness
  issues;
- the benchmark completes 1M, 2M, and 5M particles;
- no EOS+continuity run exceeds 5M particles;
- timing output reports CPU-relative speedup and hardware.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
22 passed
```

Hardware and runtime:

- host: PrediQT-02
- Python environment: PQT venv
- Python executable: `/home/kunalp/.pqt_venv_e0b41259/bin/python`
- CPU: Intel(R) Core(TM) Ultra 7 155H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.79, 8188 MiB
- Warp: 1.14.0
- PySPH: editable install from this checkout

Current capped sweep:

```text
backend particles repeats p50_ms p_checksum arho_checksum speedup_vs_cpu status
cpu_cython           1000000       1  19783.291   2.006988e+09  -3.157208e+05          1.000 ok
warp_grid_eos_cont   1000000       1    122.830   2.006988e+09  -3.157289e+05        161.063 ok
cpu_cython           2000000       1  43621.644   4.012550e+09   4.032652e+05          1.000 ok
warp_grid_eos_cont   2000000       1    318.672   4.012550e+09   4.032475e+05        136.886 ok
cpu_cython           5000000       1 130549.367   1.002678e+10   6.015853e+05          1.000 ok
warp_grid_eos_cont   5000000       1   1798.625   1.002678e+10   6.015562e+05         72.583 ok
```

Interpretation:

- Pressure checksums match to reported precision.
- Continuity aggregate checksums are close but not exact, which is expected for
  different parallel accumulation order and floating-point execution paths.
  Focused tests compare per-particle values for deterministic small fixtures.
- The 5M cap was honored. No 10M EOS+continuity run was performed.
- The Warp path now exercises both a pure per-particle equation and a
  gradient-based neighbor-loop equation.

## Conclusion

EOS and continuity are now running on the Warp path with focused correctness
coverage and large speedups through 5M particles. The next useful step is a
pressure-gradient momentum equation, which would complete the minimal density,
pressure, and acceleration chain needed for a simple WCSPH step.
