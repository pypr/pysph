---
type: experiment
id: 2026-06-15_warp-nnps-device-consumption
created: 2026-06-15T11:05:00 CET
author: @kunalpuri-prediqt
aspect: gpu-nnps
status: active
last_checked: 2026-06-15T11:20:00 CET
---

# Experiment: Warp NNPS Device Consumption

## Headline

At 1,000,000 particles on PrediQT-02, `warp_grid_reduce` reaches `145.583x`
CPU speed versus `LinkedListNNPS` for an equation-like neighbor mass sum on
Intel(R) Core(TM) Ultra 7 155H CPU and NVIDIA GeForce RTX 4060 Laptop GPU.

## Purpose

The previous NNPS baseline proved that `UniformGridWarpNNPS` can build a flat
neighbor cache on the GPU quickly. This experiment checks the next integration
question: can a kernel consume that cache directly without walking
`get_nearest_particles()` and `UIntArray` for every particle?

The current consumer is intentionally simple. It computes, for each destination
particle, the sum of a scalar source property over all neighbors. With `m=1`,
the output checksum is the total neighbor count, so CPU and GPU checksums should
match the existing NNPS count benchmark.

## Setup

Run from the repository root:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-device-consumption/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-device-consumption/run_benchmark.sh --sizes 1000000 --repeats 1
```

The wrappers use the active venv if one is already active; otherwise they
source:

```bash
source "$HOME/prediqt/activate"
```

## Hypothesis

`warp_grid_reduce` should match `cpu_reduce` for average neighbor sum and
produce a near-identical checksum while running substantially faster at
1,000,000 particles, because the GPU path builds the grid cache and consumes it
on device before returning a single checksum.

## Execution

`run_correctness.sh` runs:

```text
python -m pytest -q pysph/base/tests/test_warp_nnps.py
```

`run_benchmark.sh` runs `benchmark_neighbor_sum.py`, which records:

- CPU model;
- GPU model, driver, and memory;
- Warp version;
- backend;
- particle count;
- repeat count;
- p50 time;
- average neighbor-property sum;
- checksum;
- speedup relative to `cpu_reduce`.

## Success Criteria

This experiment succeeds when:

- focused Warp NNPS correctness tests pass;
- `cpu_reduce` and `warp_grid_reduce` report matching average neighbor sum and
  a checksum delta small enough to classify as a boundary-sensitive floating
  point difference;
- `warp_grid_reduce` reports a useful `speedup_vs_cpu` at 1,000,000 particles;
- the result lists CPU and GPU hardware.

## Results

Focused correctness:

```text
python -m pytest -q pysph/base/tests/test_warp_nnps.py
15 passed
```

Large device-consumption benchmark:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-nnps-device-consumption/run_benchmark.sh --sizes 1000000 --repeats 1
```

Hardware and runtime:

- host: PrediQT-02
- Python environment: PQT venv
- Python executable: `/home/kunalp/.pqt_venv_e0b41259/bin/python`
- CPU: Intel(R) Core(TM) Ultra 7 155H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.79, 8188 MiB
- Warp: 1.14.0
- PySPH: editable install from this checkout

Current result:

```text
backend particles repeats p50_ms avg_neighbor_sum checksum speedup_vs_cpu
cpu_reduce         1000000       1 10212.920           25.568 25568198.000          1.000
warp_grid_reduce   1000000       1    70.152           25.568 25568204.000        145.583
```

Interpretation:

- `warp_grid_reduce` builds the device-resident uniform-grid neighbor cache,
  runs a Warp kernel that sums source `m` over each destination particle's
  neighbors, and reduces the output to one checksum.
- Average neighbor sum matches to the reported precision: `25.568`.
- The checksum delta is `6` over roughly `25.6M` accumulated neighbor
  contributions. This is small enough for the large random benchmark headline,
  but exact CPU/GPU neighbor-set parity remains covered by focused deterministic
  tests rather than inferred from this aggregate run.
- On this hardware, the device-consumption path reaches `145.583x` CPU speed.

## Conclusion

The Warp grid cache is now useful beyond construction: a small equation-like
kernel can consume it entirely on device. The next implementation target is to
turn this proof into a reusable equation-loop contract instead of a one-off
neighbor-sum helper.
