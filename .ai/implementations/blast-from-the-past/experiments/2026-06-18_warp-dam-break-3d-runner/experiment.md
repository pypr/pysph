---
type: experiment
id: 2026-06-18_warp-dam-break-3d-runner
created: 2026-06-19T10:00:00 CEST
author: @kunalpuri-prediqt
agent: claude
aspect: validation-benchmarks
adr: ADR-0005
status: complete
last_checked: 2026-06-20T18:58:02 CEST
---

# Experiment: Warp 3D dam-break (Lobovsky no-obstacle)

## Headline

Second validation case for the Warp backend, and the first **3D**, **gravity-
driven**, **multi-array (fluid + solid wall)** one: the Lobovsky no-obstacle
dam-break. The additive Warp dam-break step matches both a hand-rolled CPU
baseline (tier-1) and the **real shipped PySPH `dam_break_3d_lobovsky.py`
Application** (tier-2) to fp32-vs-fp64 precision on every observable that is not
near-rest-pressure; near-rest pressure is at the fp32 Tait-EOS cancellation
floor and recovers to ~1% relative once the flow develops.

## Purpose

Exercise the four WCSPH features the 2D elliptical drop never used --
WendlandQuintic kernel, gravity, multi-array stepping, and fixed solid walls
(`TaitEOSHGCorrection`) -- end to end against the PySPH CPU references, using
the *same* initial condition the reference builds:

```text
DamBreak3DGeometry(no obstacle)  ->  [fluid, wall]
UniformGridWarpNNPS(dim=3, [fluid, wall])
wc_sph_dam_break_step            # EPEC, WendlandQuintic, Tait + Tait-HG walls
```

## Reference (CPU)

`pysph/examples/dam_break/dam_break_3d_lobovsky.py`: `WCSPHScheme` +
`EPECIntegrator` + `WendlandQuintic(dim=3)`, container ~5.367 x 0.5 x 1.5,
fluid column 2.0 x 0.5 x 1.0, `dx=H/30`, `hdx=1.3`, `rho0=1000`, `gamma=7`,
`alpha=0.25`, `beta=0`, `gz=-9.81`, `hg_correction=True`, `tf=2.5`, `n_damp=50`.

## Active physics / integration

- **WendlandQuintic** kernel (new device id 2), `radius_scale=2.0` (C2 support
  `q<2`).
- **Tait EOS** on the fluid; **`TaitEOSHGCorrection`** (clamp `rho>=rho0` so
  `p>=0`) on the wall.
- **Pressure gradient + Monaghan artificial viscosity** summed over
  `[fluid, wall]`; **continuity** for the fluid over `[fluid, wall]` and for the
  wall over `[fluid]` only.
- **XSPH** position correction from fluid neighbours only.
- **Gravity** added to the fluid acceleration (full strength).
- **Fixed walls**: walls start at rest with zero acceleration, so the shared
  PEC stage leaves their position/velocity unchanged while their density (hence
  pressure) responds to approaching fluid.
- **Adaptive timestep** with PySPH's `n_damp` startup **timestep** damping (see
  parity notes).
- Grid-direct neighbour traversal throughout; all additions are additive to the
  backend (the 2D elliptical-drop path is untouched).

## Validation tiers (all gate on `all_finite`)

### Tier 0 -- runner smoke (`run_correctness.sh`)

`dam_break_3d_runner.py --dx 0.1 --steps 20`: 1000 fluid + 3824 wall particles
built from the geometry; `all_finite: true`; walls fixed; `c0 = 32.85`
(reference scheme value); gravity pulls the fluid down (`w_mean < 0`);
`wall_p_min = 0` (HG clamp); `rho ~ rho0` (near rest at this short horizon).

### Tier 1 -- hand-rolled CPU parity (`compare_warp_pysph_dam_break_3d.py`)

A CPU EPEC reimplementation (`LinkedListNNPS(dim=3)` + `WendlandQuintic(dim=3)`)
that mirrors `wc_sph_dam_break_step` block-for-block, advanced beside Warp with a
**fixed dt, no damping, full gravity** for a clean fp32-vs-fp64 diff
(`--dx 0.15 --steps 3 --dt 1e-4`, 234 fluid + 1809 wall):

| field group | result |
| --- | --- |
| x, y, z, u, v, w, rho (fluid); rho (wall) | match to **~1e-8** relative |
| pressure (fluid, wall) | abs diff **0.17 Pa** vs the fp32 Tait floor **2.06 Pa** -> PASS |

Pressure is gated on an **absolute** tolerance derived from the fp32 Tait-EOS
cancellation floor `rho0*c0^2*2^-23` (see parity notes), not a relative one.

### Tier 2 -- resolved parity vs the real PySPH Application (`resolved_dam_break_3d_comparison.py`)

Subprocesses `dam_break_3d_lobovsky.py` (coarse `--dx 0.12 --tf 0.05`), loads the
`t~0` dump as a shared IC, and steps Warp to each CPU checkpoint time
(512 fluid + 2647 wall). Signed `warp - cpu` deltas:

| observable | t=0.0017 (cp0) | t=0.05 (cp3, developed) |
| --- | --- | --- |
| per-particle x / z (max abs) | 1.0e-7 / 6.6e-8 | -- |
| kinetic energy (rel) | 1.7e-5 | 1.9e-4 |
| surge-front x (rel) | 5e-8 | 8.6e-7 |
| max height (abs, m) | 4e-8 | 3.5e-6 |
| rho_max (rel) | 1.3e-7 | 1.8e-6 |
| **p_max (rel)** | **~13%** (near rest) | **~1.1%** (developed) |
| wall p_max (rel) | ~13% | ~1.3% |

This is the key result: kinematics, energy, surge front, and density agree to
fp32 across the whole short horizon; pressure starts at the fp32 cancellation
floor (near rest) and **recovers to ~1% relative once the column collapse builds
a real pressure field** -- exactly the tier-1 prediction.

## Parity notes (honest deltas vs the reference)

- **fp32 Tait-EOS cancellation floor.** Near rest `rho ~ rho0`, so
  `p = B*((rho/rho0)^gamma - 1)` is a tiny difference of near-equal quantities.
  fp32 stores `rho ~ 1000` with absolute error `~rho0*2^-23 ~ 1.2e-4`, and
  `dp/drho ~ B*gamma/rho0 ~ 1079`, so `dp ~ 0.13 Pa` per particle -- matching the
  observed ~0.17 Pa. The *relative* p error is large only because p itself is
  ~1 Pa near rest; in *absolute* terms pressure matches, and relative agreement
  recovers as p grows. Tier-1 gates pressure on this absolute floor.
- **EPEC, not "PEC vs EPEC".** `wc_sph_dam_break_step` evaluates accelerations,
  predicts to the half step, re-evaluates, then corrects (E-P-E-C). That matches
  the reference `EPECIntegrator`, so the ADR-0005 "EPEC vs PEC" risk is resolved
  -- the reference is run with its native `EPECIntegrator`.
- **`n_damp` damps the timestep, not gravity.** PySPH `Solver._damp_timestep`
  scales `dt` by `0.5*(sin(pi*(-0.5 + (count+1)/n_damp)) + 1)` over the first
  `n_damp` steps; it does **not** ramp gravity. The runner therefore applies that
  factor to the adaptive `dt` (via `adaptive_dt_scale`) with full gravity. The
  additive `gravity_ramp` backend feature exists but is left at 1.0 here. (ADR-0005
  step 3 described a gravity ramp; the PySPH-faithful choice is timestep damping.)
- **`c0` reference inconsistency.** The reference builds its `WCSPHScheme` with
  the module constant `c0 = 10*sqrt(2*9.81*0.55) ~ 32.85` (the physics sound
  speed) but computes its initial `dt` from `10*get_max_speed = 10*sqrt(2*9.81*H)
  ~ 44.29`. The runner matches the physics value (32.85) for `c0` and uses the
  44.29 form only for the initial `dt` cap, exactly as the reference does.
- **Cache stability (ADR-0005, precise statement).** The *generated* group source
  for the cubic/gaussian 2D kernels is **byte-identical** (kernel choice is a
  runtime `kernel_id`, not source) and is pinned by
  `test_2d_path_generated_source_is_byte_identical_to_golden`. The cubic (id 0)
  and gaussian (id 1) numerical code paths in the shared `_kernel_value/dwdq`
  routers are **unchanged** (only an additive `id==2` branch). The router source
  itself grew by that branch, so a one-time, **logic-preserving** recompile of
  the warm cache can occur; it does not perturb the committed 2D baseline. (This
  refines the ADR's "on-disk cache byte-identical" wording -- see the ADR update.)

## How to run

```bash
PY=/home/kunalp/.pqt_venv_e0b41259/bin/python   # venv with warp+pysph
# smoke
ROOT=$(git rev-parse --show-toplevel) bash <pkt>/run_correctness.sh
# tier-1 (hand-rolled CPU parity)
$PY <pkt>/compare_warp_pysph_dam_break_3d.py --dx 0.15 --steps 3 --dt 1e-4
# tier-2 (real PySPH Application)
$PY <pkt>/resolved_dam_break_3d_comparison.py --dx 0.12 --tf 0.05 --pfreq 15
```

Performance + representative snapshot (real PySPH CPU vs Warp fp32, same `tf`):

```bash
$PY <pkt>/perf_and_snapshot_dam_break_3d.py --dx 0.08 --tf 0.4
```

At `dx=0.08` (7,392 particles), RTX 4060 fp32 vs single-threaded PySPH Cython
fp64, `tf=0.4 s`: per-step **8.5 ms (Warp) vs 13.0 ms (CPU) = ~1.53x**, and total
wall **3.92 s vs 6.01 s = ~1.53x** -- per-step and wall agree because the
adaptive-dt schedules match (Warp 460 steps vs CPU 461). Note: the reference's
`dt = 0.25*h0/(1.1*c_max)` is only the *seed* dt; PySPH grows dt to the
CFL-limited value with no clamp, so the runner must NOT cap at the seed
(`dt_max` defaults to uncapped -- CFL + `n_damp` govern). This is the small-N /
overhead-bound regime; the committed cross-GPU sweep measured ~57.6x per-step at
1M particles on the same 4060.

The snapshot below (x-z, fluid by speed, walls grey, `t=0.4 s`) shows the two
runs visually indistinguishable:

![CPU fp64 vs Warp fp32 dam-break snapshot at t=0.4 s](cpu-vs-warp-snapshot.png)

### Large-N (>1M particles) and a 3D-explicit snapshot

`bench_1M_and_3d_snapshot.py` -- per-step throughput at scale (a developed 1M run
to a physical `tf` is multi-hour on the CPU, so per-step over a fixed step count
is used, as in the committed 1M elliptical comparison):

```bash
$PY <pkt>/bench_1M_and_3d_snapshot.py --dx 0.0108 --snapshot-tf 0   # perf only
$PY <pkt>/bench_1M_and_3d_snapshot.py --dx 0.011  --snapshot-tf 0.2 # + 3D snapshot
```

At **1,014,072 particles**: Warp fp32 (fused) **0.337 s/step (3.01 M
particle-steps/s)** vs PySPH CPU fp64 **4.5-5.0 s/step (~0.21 M)** = **~13-15x
per-step** on the RTX 4060 (CPU per-step has ~10-20% run-to-run variance; Warp is
stable). The GPU advantage scales with N (1.53x at 7k -> ~14x at >1M). The fluid
acceleration+density blocks are **fused** (pressure + AV + continuity in one
kernel per source) -- this made the Warp step 1.23x faster (0.415 -> 0.337 s/step)
than the initial single-block composition. It is still below the elliptical
drop's 57.6x at 1M because XSPH (fluid-only) and wall continuity (fluid->wall)
have heterogeneous source/destination sets and stay separate, so the step is not a
single fused kernel.

The four-view snapshot at 999,975 particles (`t=0.2 s`) makes the 3D structure
explicit (x-z side, x-y top-down, y-z end, 3D scatter):

![Warp fp32 3D dam break, ~1M particles, four views at t=0.2 s](bench-1M-3d-snapshot.png)

### Cross-GPU sweep

`gpu_perf_sweep_dam_break.py` sweeps particle count (dx) per GPU; lenses +
figures in `gpu-sweep/` (`README.md`). At ~1M, fixed-dt fused step: RTX PRO 6000
Blackwell 39.1 Mp-st/s (163x vs 1 CPU core) > L40S 25.9 Mp-st/s (108x) > RTX 4060
4.5 Mp-st/s (19x). Cost/energy favour the L40S ($0.0114 vs $0.0187 per billion
p-steps; 13.5 vs 15.4 kJ/billion). (5090 + B300 deferred.)

Artifacts: `results-smoke.npz`, `comparison-tier1-summary.json` (+ warp npz),
`comparison-resolved-summary.json`, `cpu-vs-warp-perf.json`,
`cpu-vs-warp-snapshot.png`, `bench-1M-perf.json`, `bench-1Mplus-perf.json`,
`bench-1M-3d-snapshot.png`.

### Medium-resolution showcase run (2026-06-20)

Ran a developed Warp-only case at `dx=0.025`: 59,280 fluid + 66,407 wall =
125,687 total particles. A 10-step timing slice measured `0.03140 s/step`
versus `0.42733 s/step` for the single-threaded PySPH CPU baseline (`13.61x`).
The adaptive GPU run then advanced 2,837 steps to `t=0.8016 s`:

```text
all_finite: true
rho_min / rho_max: 989.7354 / 1014.1267 kg/m^3
surge_front_x: 5.08698 m
max_height: 0.84473 m
kinetic_energy: 1226.1269
wall_p_min / wall_p_max: 0.0 / 24391.78 Pa
```

Also captured the visually stronger collapse phase at 1,415 steps / `t=0.3996`
s (`all_finite=true`, surge front 3.1553 m, max height 0.9391 m). Artifacts:

- `showcase-dx025-t080-perf.json`
- `reviews/2026-06-19_warp-3d-dam-break-lobovsky_assets/showcase-dx025-t080-3d-snapshot.png`
  (four-view verification figure, embedded in the review)
- `reviews/2026-06-19_warp-3d-dam-break-lobovsky_assets/showcase-dx025-t040-hero.png`
  (PyVista hero frame, particles coloured by speed, embedded in the review)

Splashsurf surface reconstruction was exercised successfully, but Blender and
ffmpeg are not installed on this host; large intermediate PLY meshes were not
retained. The hero is an honest particle visualization, not a photorealistic
render.

## What to expect

- non-empty fluid + wall arrays, `all_finite: true`, walls fixed;
- tier-1 `passed: true` (kinematics/density relative, pressure absolute floor);
- tier-2 fp32-scale deltas on KE / surge front / height / density, and pressure
  recovering to ~1% relative as the flow develops.

## Out of scope (follow-ups)

- SPHERIC/Kleefsman obstacle case (third array).
- Long-horizon (`tf=2.5`) run + probe-point pressure `p/(rho g H)` vs
  `db_exp_data.get_lobovsky_data()` (the full experimental comparison; needs an
  SPH interpolator and a multi-second run).
- A cubic/gaussian dam-break cross-check; performance / cross-GPU
  characterisation (correctness-only here).
