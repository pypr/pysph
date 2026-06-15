---
type: experiment
id: 2026-06-15_warp-pressure-gradient
created: 2026-06-15T14:05:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: active
last_checked: 2026-06-15T14:25:00 CET
---

# Experiment: Warp Pressure Gradient

## Headline

At 5,000,000 particles on PrediQT-02, `warp_grid_pgrad` computes the inviscid
pressure-gradient operation in `2.988 s` versus CPU/Cython `115.695 s`, a
`38.722x` speedup, on Intel(R) Core(TM) Ultra 7 155H CPU and NVIDIA GeForce RTX
4060 Laptop GPU.

## Purpose

Add the minimal pressure-gradient acceleration kernel needed after density,
EOS, and continuity:

```text
a_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad(W_ij)
```

This is the inviscid pressure portion of PySPH's WCSPH momentum equation,
without artificial viscosity, tensile correction, CFL bookkeeping, or body
force. Those pieces should be added separately so each part has clean tests and
benchmarks.

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-pressure-gradient/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-pressure-gradient/run_benchmark.sh --sizes 1000000 2000000 5000000 --repeats 1
```

The wrappers use the active venv if one is already active; otherwise they
source:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

The Warp pressure-gradient kernel should match CPU per-particle reference
fixtures and provide large speedups for million-particle operation benchmarks,
while staying capped at 5M particles.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
```

`run_benchmark.sh` runs `benchmark_pressure_gradient.py`, which records:

- CPU model;
- GPU model, driver, and memory;
- Warp version;
- backend;
- particle count;
- repeat count;
- p50 operation time;
- acceleration checksums;
- speedup relative to `cpu_cython`.

CPU baseline:

- PySPH `SPHEvaluator`;
- Cython backend;
- custom `PressureGradientOnly` equation containing the same inviscid pressure
  operation;
- `CubicSpline(dim=2)`;
- `LinkedListNNPS`.

Warp baseline:

- `compute_pressure_gradient()`;
- `UniformGridWarpNNPS` device neighbor cache;
- Warp CubicSpline gradient kernel.

## Success Criteria

This experiment succeeds when:

- focused tests compare Warp pressure-gradient values against CPU references;
- the benchmark completes 1M, 2M, and 5M particles;
- no pgrad run exceeds 5M particles;
- timing output reports CPU-relative speedup and hardware.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
24 passed
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
backend particles repeats p50_ms au_checksum av_checksum aw_checksum speedup_vs_cpu status
cpu_cython        1000000       1  18838.161  -5.329071e-15   3.552714e-15   0.000000e+00          1.000 ok
warp_grid_pgrad   1000000       1    126.530  -6.198883e-06  -7.152557e-06   0.000000e+00        148.884 ok
cpu_cython        2000000       1  42210.247  -2.842171e-14   1.421085e-14   0.000000e+00          1.000 ok
warp_grid_pgrad   2000000       1    325.059  -1.096725e-05   2.288818e-05   0.000000e+00        129.854 ok
cpu_cython        5000000       1 115694.749  -9.947598e-14   0.000000e+00   0.000000e+00          1.000 ok
warp_grid_pgrad   5000000       1   2987.800   2.288818e-05   0.000000e+00   0.000000e+00         38.722 ok
```

Interpretation:

- Same-array total acceleration checksums are near zero because the pressure
  interaction is pair-symmetric; focused per-particle tests carry the primary
  correctness signal.
- The 5M cap was honored. No 10M pressure-gradient run was performed.
- The Warp path now has the density, pressure, continuity, and inviscid
  acceleration pieces needed for a minimal GPU dynamics step.

## Conclusion

The inviscid pressure-gradient kernel is now running on Warp and benchmarked
through 5M particles. The next useful step is a tiny Euler or PEC-style
integrator and a short drop-like GPU loop that wires NNPS, density, EOS,
pressure gradient, and position/velocity update together.
