# Producing the production results report (run on a capable GPU machine)

The headline runs (especially the million-particle CPU PySPH Application and any
longer resolved run) are CPU-heavy and best run on a capable machine. This
directory ships a driver that runs them fresh and assembles a Markdown report.

## 1. Set up the environment

Follow [`BUILD.md`](../../../../../BUILD.md) at the repo root: build the Cython
extensions and install `warp-lang`. Confirm the GPU is detected:

```bash
python -c "import warp as wp; wp.init()"   # should list your CUDA device
```

GPU compatibility (V100/A100/H100/RTX 5090/RTX PRO 6000 Blackwell) is covered in
BUILD.md section 10 -- no code changes are needed per GPU.

## 2. Validate the pipeline (tiny, ~1-2 min)

```bash
R=.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner
PYTHONPATH=$R python $R/generate_results_report.py --quick \
    --out-dir /tmp/report-smoke --gpu-label "smoke"
```

This runs nx=20 resolved + nx=30 / 5-step million and writes
`/tmp/report-smoke/RESULTS_REPORT.md`. If that looks right, run the full report.

## 3. Full report

```bash
PYTHONPATH=$R python $R/generate_results_report.py \
    --out-dir results-report --gpu-label "NVIDIA H100 80GB"
```

- Runs the **resolved nx=100 apples-to-apples** comparison (real PySPH CPU
  Application vs Warp, identical step count) and the **million-particle /
  100-fixed-step** throughput comparison.
- For a fuller trajectory comparison add e.g. `--resolved-tf 0.0008,0.0038`
  (default) or a larger final time -- the longer CPU run is the multi-hour part
  that this machine is for.
- Writes `results-report/RESULTS_REPORT.md` plus the raw summary JSONs.

## Cross-GPU performance sweep

To compare GPUs, run `gpu_perf_sweep.py` on each one and paste the JSON block it
prints. It sweeps particle count (via `nx`) and reports the steady per-step wall
time and throughput (particle-steps/s) for the grid-direct continuity PEC step;
it's GPU-only (fast), warms the kernel once, and catches OOM per point so it
finds the capacity ceiling.

```bash
PYTHONPATH=$R python $R/gpu_perf_sweep.py --label "<GPU name>" --output sweep-<gpu>.json
# custom: --nx-list 100,200,400,565,1000,1400,1800,2600 --steps 16 --warmup 6
```

`nx` maps to particle count via the disk fill (~`pi*nx^2`): nx=100 ~ 31k,
565 ~ 1.0M, 1000 ~ 3.1M, 1800 ~ 10M. The collected per-GPU JSONs become the
cross-GPU performance artifact.

## Notes

- The CPU baseline is the real single-threaded PySPH Cython Application; report
  it as GPU-vs-single-threaded-CPU.
- The million-particle run uses fixed timesteps so both sides do identical work.
- Memory headroom (A100 80 GB / H100 / 96 GB Blackwell) allows scaling the
  particle count well beyond the 8 GB dev box; raise `--million-nx` accordingly
  (watch host RAM for the initial numpy mgrid).
- First Warp run on a new GPU cold-compiles the fused grid kernel once, then
  loads from the on-disk cache.
