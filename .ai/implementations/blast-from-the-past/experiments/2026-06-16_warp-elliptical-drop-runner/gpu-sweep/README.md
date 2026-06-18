# Cross-GPU performance sweep -- Warp grid-direct WCSPH

Per-GPU throughput sweeps for the grid-direct continuity-density PEC step (fp32),
produced by `../gpu_perf_sweep.py`. Run the script on each GPU and drop its JSON
here; this doc collects the curves and a cross-GPU comparison.

**Metric:** `throughput = particles / per_step_median` (particle-steps per
second), the steady warm per-step over the elliptical-drop workload. Physics:
Gaussian kernel, Tait EOS, continuity density, radius_scale=3, fixed dt, fp32
(`compyle use_double=False`). nx maps to particle count via the disk fill
(`particles ~ pi*nx^2`).

## GPUs collected

| GPU | arch | VRAM | data | file |
|---|---|---:|---|---|
| NVIDIA L40S | sm_89 (Ada) | 48 GiB | full sweep (31k-21M) | `sweep-l40s.json` |
| NVIDIA RTX PRO 6000 Blackwell | sm_120 (Blackwell) | 96 GiB | 1M point only* | `../million-cpu-gpu-grid-direct/blackwell-rtxpro6000-million-100step-summary.json` |
| NVIDIA RTX 4060 Laptop | sm_89 (Ada) | 8 GiB | 1M point only* | `../million-cpu-gpu-grid-direct/fresh-headline-speedups.json` |

\* full `gpu_perf_sweep.py` runs for Blackwell and the 4060 are pending; only the
single 1M-particle point is on record for those so far.

## Full sweep -- NVIDIA L40S (sm_89, 44 GiB usable)

| nx | particles | per-step (s) | throughput (particle-steps/s) | finite |
|---:|---:|---:|---:|:--:|
| 100 | 31,417 | 0.001509 | 2.08e7 | yes |
| 200 | 125,629 | 0.002607 | 4.82e7 | yes |
| 400 | 502,625 | 0.006209 | 8.10e7 | yes |
| 565 | 1,002,885 | 0.010915 | 9.19e7 | yes |
| 1000 | 3,141,549 | 0.032157 | 9.77e7 | yes |
| 1400 | 6,157,477 | 0.063098 | 9.76e7 | yes |
| 1800 | 10,178,545 | 0.102492 | 9.93e7 | yes |
| 2600 | 21,236,953 | 0.356544 | 5.96e7 | yes |

- Throughput ramps with particle count (better GPU saturation) and **plateaus at
  ~9.8e7 particle-steps/s from ~1M to ~10M**, with per-step scaling roughly
  linearly (1M -> 10M is ~9.4x particles for ~9.4x time).
- At **21M** throughput drops to 5.96e7 (per-step 3.5x higher for 2.1x more
  particles) -- a super-linear knee, not OOM (44 GiB has headroom). Likely grid
  build / cache-locality at very large N; worth profiling later.
- All points finite. Cold compile on this fresh box was ~89 s
  (warp_nnps 20.9 s + the generated grid kernel 66.2 s), then disk-cached.

## Cross-GPU comparison at 1M particles (nx=565)

| GPU | per-step (s) | throughput (particle-steps/s) | vs RTX 4060 |
|---|---:|---:|---:|
| RTX 4060 Laptop | 0.0598 | 1.68e7 | 1.0x |
| L40S | 0.010915 | 9.19e7 | 5.5x |
| RTX PRO 6000 Blackwell | 0.008625 | 1.16e8 | 6.9x |

Note: 1M is small enough that Blackwell is only ~1.26x the L40S here; a full
Blackwell sweep would likely show a higher plateau (more headroom at larger N).
The L40S plateau (~9.8e7) is the representative steady throughput for that card.
