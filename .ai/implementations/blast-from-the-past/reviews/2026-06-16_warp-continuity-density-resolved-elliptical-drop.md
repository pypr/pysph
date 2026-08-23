---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/plans/2026-06-16_warp-continuity-density-leapfrog-parity.md
adrs: []
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: lgtm
---

# Review: Warp Continuity-Density Resolved Elliptical Drop

## Diff summary

- Covers both the continuity-density parity plan and the preceding resolved
  comparison plan:
  `.ai/implementations/blast-from-the-past/plans/2026-06-16_warp-resolved-elliptical-drop-performance-comparison.md`.
- Added a Warp continuity-density WCSPH PEC-style path behind
  `wc_sph_leapfrog_step(..., density_mode='continuity')`.
- Preserved the existing summation-density KDK path as the default behavior.
- Added device kernels/helpers to save reference state (`x0/y0/z0`,
  `u0/v0/w0`, `rho0`) and apply staged updates to position, velocity, and
  density from `arho`.
- Threaded `density_mode` through the Warp elliptical-drop runner and the
  resolved PySPH Application comparison script.
- Added resolved `nx=100` CPU/Warp comparison summaries and side-by-side PNGs.
- Updated experiment, aspect, daily, current, and session-log memory.

## Behavioral / numerical changes

- `density_mode='continuity'` now matches the PySPH no-scheme Application's
  density formulation: Tait EOS from current `rho`, `ContinuityEquation`
  computes `arho`, and `rho` is advanced from saved `rho0` through PEC-style
  stages.
- The original resolved comparison using summation density is explicitly
  diagnostic only. It explained the previous 2225/4807-step Warp behavior:
  summation-density refreshes created larger density/pressure excursions and
  collapsed the adaptive force timestep.
- The continuity resolved `nx=100` run reached both requested checkpoint times:
  PySPH CPU Application took `228.25765374601178` s / 1393 steps; Warp took
  `30.008050591000938` s / 1804 steps; wall-time speedup was
  `7.606547218180963x`.
- At `t=0.0038`, CPU-vs-Warp deltas were:
  major axis `1.5947661098358878e-06`, minor axis
  `2.1943316564909665e-06`, `rho_min -4.816405699936688e-06`,
  `rho_max 8.755722542552746e-07`, kinetic energy
  `-0.00043376772100600647`.
- The remaining step-count difference appears tied to timestep policy and the
  resolved runner's explicit cap to hit output times, not to a density
  formulation mismatch.

## Validation

```text
python -m py_compile pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py
pass
```

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py -q -k 'pec_stage or continuity_mode'
2 passed, 2 warnings in 11.98s
```

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
38 passed, 2 warnings in 4.12s
```

```text
python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/warp_elliptical_drop_runner.py --nx 8 --steps 2 --dt 1.0e-5 --rho0 1.0 --c0 20.0 --p0 0.0 --alpha 0.1 --beta 0.0 --gamma 7.0 --adaptive-dt --cfl 0.25 --dt-min 1.0e-7 --density-mode continuity
all_finite: true
density_mode: continuity
rho_min: 0.9995885491371155
rho_max: 1.0004163980484009
```

```text
python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-continuity --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved --max-steps 10000000
PySPH CPU Application: 228.25765374601178 s, 1393 steps
Warp continuity-density: 30.008050591000938 s, 1804 steps
speedup_wall_time: 7.606547218180963
t=0.0008 major-axis delta: 3.6375168877000874e-08
t=0.0038 major-axis delta: 1.5947661098358878e-06
t=0.0038 rho_min delta: -4.816405699936688e-06
```

```text
python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

```text
git diff --check -- .ai/implementations/blast-from-the-past pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py
pass
```

## Visual aid

- `resolved-nx100-continuity-t0p0008000.png`
- `resolved-nx100-continuity-t0p0038000.png`

Visual inspection of
`.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-continuity-t0p0038000.png`
shows CPU and Warp particle clouds and exact ellipse overlays are visually
indistinguishable at the final checkpoint.

## Risks / unresolved questions

- The continuity path is PEC-style, while the public helper name remains
  `wc_sph_leapfrog_step()` for continuity with earlier prototype wiring. This
  is documented via `density_mode`.
- Warp still caps `dt` to land exactly on checkpoint times in the resolved
  comparison runner; PySPH's output-time handling is not yet matched exactly.
- True periodic neighbor interactions remain out of scope for this slice.
- Raw HDF5/NPZ checkpoint dumps were generated locally and pruned from the
  reviewable artifact set; summary JSON and PNG comparisons remain.
- Unrelated untracked `CODEBASE_UNDERSTANDING.md` remains untouched.

## Reviewer verdict

- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-17T00:36:00 CEST
