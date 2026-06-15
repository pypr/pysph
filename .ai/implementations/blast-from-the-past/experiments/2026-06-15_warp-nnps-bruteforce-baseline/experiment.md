---
type: experiment
id: 2026-06-15_warp-nnps-bruteforce-baseline
created: 2026-06-15T10:05:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: active
last_checked: 2026-06-15T10:45:00 CET
---

# Experiment: Warp NNPS Baselines

## Headline

At 1,000,000 particles on PrediQT-02, `warp_grid_device` reaches `88.288x`
CPU speed versus `LinkedListNNPS` while matching the average neighbor count
(`25.568`) on Intel(R) Core(TM) Ultra 7 155H CPU and NVIDIA GeForce RTX 4060
Laptop GPU.

## Purpose

Establish the first Warp NNPS implementation baselines.

The implementation uses Warp kernels for the geometric distance test and
returns source-local neighbor indices through PySPH's existing `UIntArray`
contract. It supports a per-query flags path, a cached flat-neighbor path, and
a uniform-grid/cell-list path.

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh
```

The wrappers use the active venv if one is already active; otherwise they
source:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

Warp brute force should match CPU linked-list NNPS neighbor sets for small,
deterministic fixtures covering:

- 1D, 2D, and multiple particle arrays;
- source/destination smoothing-length inclusion;
- sorted-gid ordering;
- update after ParticleArray mutation.

The uncached brute-force implementation is expected to have low CPU-relative
speedup for many queries because it launches a kernel and reads flags back for
each destination particle. The cached brute-force path avoids per-particle
launches but remains O(N^2). The uniform-grid path builds source cell lists and
scans adjacent cells, making it the first performance-relevant Warp NNPS
baseline.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_nnps.py
```

`run_benchmark.sh` runs `benchmark_warp_nnps.py`, which records:

- CPU model;
- GPU model, driver, and memory;
- Warp version;
- backend;
- particle count;
- repeat count;
- all-particle query p50 time;
- average neighbor count;
- speedup relative to CPU.

For large particle counts, avoid the uncached brute-force backends. Use
`warp_grid` for PySPH host-facing neighbor access and `warp_grid_device` for
bulk GPU cache construction:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh --sizes 1000000 --repeats 1 --backends cpu warp_grid
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh --sizes 1000000 --repeats 1 --backends cpu warp_grid_device
```

## Success Criteria

This experiment succeeds when:

- all focused Warp NNPS correctness tests pass;
- benchmark cases complete without changing neighbor counts;
- timing output separates CPU linked-list, uncached Warp brute-force, and cached
  Warp brute-force, and Warp grid paths;
- timing output reports CPU-relative speedup;
- sub-1.0 Warp speedups are classified as expected per-query launch/readback
  cost or follow-up optimization targets.

## Results

Initial focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_nnps.py
12 passed
```

Current smoke benchmark:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh --sizes 128 --repeats 1
```

Hardware and runtime:

- host: PrediQT-02
- Python environment: PQT venv
- Python executable: `/home/kunalp/.pqt_venv_e0b41259/bin/python`
- CPU: Intel(R) Core(TM) Ultra 7 155H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.79, 8188 MiB
- Warp: 1.14.0
- PySPH: editable install from this checkout

Expected/current smoke result:

```text
backend particles repeats p50_ms avg_neighbors speedup_vs_cpu
cpu                128       1    0.436        21.781          1.000
warp               128       1  150.225        21.781          0.003
warp_cached        128       1   22.861        21.781          0.019
warp_grid          128       1   10.532        21.781          0.041
```

Interpretation:

- CPU, uncached Warp, cached Warp, and Warp grid report the same average
  neighbor count for the smoke case.
- `speedup_vs_cpu` is `cpu_p50_ms / backend_p50_ms`; values below `1.0` mean
  the backend has not yet reached CPU speed.
- Uncached Warp brute force is slower here because it launches one kernel per
  destination particle and reads a source-length flags array back to host for
  every query.
- Cached Warp brute force builds a flat neighbor list and is substantially
  faster than the uncached path, but it remains an O(N^2) bridge.
- Warp grid builds source cell lists and scans adjacent cells. It is now the
  first cell-list baseline, though the smoke run still includes host-facing
  neighbor-list materialization.
- On this 128-particle smoke case, the best Warp path is `warp_grid` at
  `0.041x` CPU speed. This is a baseline for optimization, not a claimed
  acceleration result.

Large host-facing benchmark:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh --sizes 1000000 --repeats 1 --backends cpu warp_grid
```

Hardware and runtime are the same as above.

```text
backend particles repeats p50_ms avg_neighbors speedup_vs_cpu
cpu            1000000       1 17253.885        25.568          1.000
warp_grid      1000000       1  4041.391        25.568          4.269
```

Interpretation:

- At 1,000,000 particles, CPU and `warp_grid` report the same average neighbor
  count.
- `warp_grid` reaches `4.269x` CPU speed on the current host-facing benchmark.
- This result still includes PySPH-style host-facing neighbor access, so a
  device-resident equation-consumption path remains the next performance target.

Large device-oriented benchmark:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-bruteforce-baseline/run_benchmark.sh --sizes 1000000 --repeats 1 --backends cpu warp_grid_device
```

Hardware and runtime are the same as above.

```text
backend particles repeats p50_ms avg_neighbors speedup_vs_cpu
cpu            1000000       1  6236.716        25.568          1.000
warp_grid_device   1000000       1    70.640        25.568         88.288
```

Interpretation:

- At 1,000,000 particles, CPU and `warp_grid_device` report the same average
  neighbor count.
- `warp_grid_device` reaches `88.288x` CPU speed for bulk grid neighbor-cache
  construction.
- This is the relevant GPU-side result. It avoids the per-particle
  `get_nearest_particles()` loop and does not materialize every particle's
  neighbors through `UIntArray`.
- The remaining integration target is to let equation kernels consume this
  device-resident neighbor cache directly.

## Conclusion

Warp NNPS has a first correctness-oriented implementation, a cached flat
neighbor-list bridge, and a uniform-grid/cell-list baseline. The next useful
implementation work is optimizing the grid path and reducing host readback.

## Follow-ups

- Optimize uniform-grid/cell-list update and query kernels.
- Keep cell-list results device-resident for equation-kernel consumption.
- Add periodic boundary fixtures before Application integration.
