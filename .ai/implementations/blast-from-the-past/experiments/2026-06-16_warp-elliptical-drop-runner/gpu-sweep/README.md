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
| NVIDIA RTX 5090 | sm_120 (Blackwell consumer) | 32 GiB | full sweep (31k-21M) | `sweep-rtx5090.json` |
| NVIDIA RTX PRO 6000 Blackwell | sm_120 (Blackwell workstation) | 96 GiB | full sweep (31k-40M) | `sweep-rtxpro6000.json` |
| NVIDIA B300 SXM6 | sm_103 (Blackwell Ultra) | 268 GiB | full sweep (31k-78M) | `sweep-b300.json` |
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

## Full sweep -- NVIDIA RTX 5090 (sm_120 consumer Blackwell, 31 GiB)

| nx | particles | per-step (s) | throughput (particle-steps/s) | finite |
|---:|---:|---:|---:|:--:|
| 100 | 31,417 | 0.002473 | 1.27e7 | yes |
| 200 | 125,629 | 0.003915 | 3.21e7 | yes |
| 400 | 502,625 | 0.009707 | 5.18e7 | yes |
| 565 | 1,002,885 | 0.014879 | 6.74e7 | yes |
| 1000 | 3,141,549 | 0.043096 | 7.29e7 | yes |
| 1400 | 6,157,477 | 0.084586 | 7.28e7 | yes |
| 1800 | 10,178,545 | 0.221873 | 4.59e7 | yes |
| 2600 | 21,236,953 | 0.511236 | 4.15e7 | yes |

- Plateaus lower (**~7.3e7** from ~1M to ~6M) than the L40S, and the
  super-linear knee comes **earlier, at ~10M** (vs the L40S's 21M) -- consistent
  with the 5090's smaller 31 GiB (capacity pressure sooner) and consumer-card
  power/clock limits. Same `sm_120` arch as the workstation Blackwell, but
  capacity/bandwidth-shaped throughput on this fp32 SPH workload.

## Full sweep -- NVIDIA RTX PRO 6000 Blackwell (sm_120 workstation, 95 GiB)

| nx | particles | per-step (s) | throughput (particle-steps/s) | finite |
|---:|---:|---:|---:|:--:|
| 100 | 31,417 | 0.001683 | 1.87e7 | yes |
| 200 | 125,629 | 0.002368 | 5.30e7 | yes |
| 400 | 502,625 | 0.005003 | 1.00e8 | yes |
| 565 | 1,002,885 | 0.008223 | 1.22e8 | yes |
| 1000 | 3,141,549 | 0.021753 | **1.44e8** | yes |
| 1400 | 6,157,477 | 0.098554 | 6.25e7 | yes |
| 1800 | 10,178,545 | 0.163877 | 6.21e7 | yes |
| 2600 | 21,236,953 | 0.396781 | 5.35e7 | yes |
| 3600 | 40,714,821 | 0.790790 | 5.15e7 | yes |

Peak **1.44e8** at 3M, then a sharp knee at 6M (per-step 0.022 -> 0.099 s).

## Full sweep -- NVIDIA B300 SXM6 (sm_103 Blackwell Ultra, 268 GiB)

| nx | particles | per-step (s) | throughput (particle-steps/s) | finite |
|---:|---:|---:|---:|:--:|
| 100 | 31,417 | 0.001025 | 3.07e7 | yes |
| 200 | 125,629 | 0.001594 | 7.88e7 | yes |
| 400 | 502,625 | 0.003614 | 1.39e8 | yes |
| 565 | 1,002,885 | 0.007132 | 1.41e8 | yes |
| 1000 | 3,141,549 | 0.022015 | **1.43e8** | yes |
| 1400 | 6,157,477 | 0.043366 | 1.42e8 | yes |
| 1800 | 10,178,545 | 0.128046 | 7.95e7 | yes |
| 2600 | 21,236,953 | 0.305089 | 6.96e7 | yes |
| 3600 | 40,714,821 | 0.583046 | 6.98e7 | yes |
| 5000 | 78,539,677 | 1.120131 | 7.01e7 | yes |

Holds the **~1.43e8 plateau from ~0.5M to 6M**, knee at 10M, then a stable
~7.0e7 post-knee plateau out to **78.5M particles** (largest run; ~1.12 s/step).

## Cross-GPU comparison

At **1M particles (nx=565)** -- the common anchor:

| GPU | arch | VRAM | per-step (s) | throughput (p-steps/s) | vs RTX 4060 |
|---|---|---:|---:|---:|---:|
| RTX 4060 Laptop | sm_89 | 8 GiB | 0.0598 | 1.68e7 | 1.0x |
| RTX 5090 | sm_120 | 32 GiB | 0.014879 | 6.74e7 | 4.0x |
| L40S | sm_89 | 48 GiB | 0.010915 | 9.19e7 | 5.5x |
| RTX PRO 6000 Blackwell | sm_120 | 96 GiB | 0.008223 | 1.22e8 | 7.3x |
| B300 SXM6 | sm_103 | 268 GiB | 0.007132 | 1.41e8 | 8.4x |

Peak / sustained throughput and the super-linear knee:

| GPU | peak throughput | peak at | knee at | post-knee plateau | max run |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | ~7.3e7 | 3M | 10M | ~4.2e7 | 21M |
| L40S | ~9.9e7 | 10M | 21M | -- | 21M |
| RTX PRO 6000 | ~1.44e8 | 3M | 6M | ~5.2e7 | 41M |
| B300 SXM6 | ~1.43e8 | 1M-6M | 10M | ~7.0e7 | 78M |

Observations (`*` = open question):
- **Both Blackwell cards hit the same peak ceiling (~1.43e8)** despite very
  different class/VRAM -- the step looks bandwidth/occupancy-bound on Blackwell,
  not raw-FLOP-bound. The **B300's edge is scale**: it sustains the peak to 6M,
  reaches **78.5M particles**, and holds a higher post-knee plateau (~7.0e7 vs
  RTX PRO 6000 ~5.2e7, 5090 ~4.2e7).
- On this fp32 SPH workload the **datacenter Ada L40S (9.19e7) beats the consumer
  5090 (6.74e7)** at 1M -- capacity/bandwidth/occupancy shaped, not FLOP shaped.
- `*` The **super-linear knee is non-monotonic with VRAM**: RTX PRO 6000 (95 GiB)
  knees at 6M, B300 (268 GiB) at 10M, 5090 (32 GiB) at 10M, L40S (44 GiB) at
  21M. So it is **not** a capacity ceiling -- likely an algorithmic/grid effect
  (cell-list build cost, occupancy, or Warp mempool behavior at scale). The
  segmented profiler (`profile_grid_direct_neighbors.py`: grid-build vs equation
  time) is the tool to diagnose it; treat the cause as unconfirmed.
- arch note: B300 reports **sm_103** (Blackwell Ultra) -- distinct from
  datacenter Blackwell sm_100 and consumer/workstation sm_120; all ran on Warp
  1.14 / CUDA Toolkit 12.9 with no code or build changes (BUILD.md section 10).
- The 4060 1M point is from the fixed-step headline run, not a `gpu_perf_sweep.py`
  sweep (close enough for the anchor; a full 4060 sweep would round out the low end).
