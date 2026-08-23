---
type: review
date: 2026-06-17
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/plans/2026-06-17_warp-adaptive-timestep-policy-parity.md
adrs: []
aspects_touched: [warp-backend, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_sph.py, pysph/base/tests/test_warp_sph.py]
status: lgtm
---

# Review: Warp Adaptive Timestep Policy Parity

## Diff summary

- Added scalar adaptive timestep policy controls to `wc_sph_leapfrog_step()`:
  `adaptive_dt_scale` and `step_dt_max`.
- Applied those controls to both the summation-density KDK path and the
  continuity-density PEC-style path after the device-reduced adaptive timestep
  candidate is computed.
- Added a focused Warp SPH test for adaptive timestep scaling and current-step
  capping.
- Updated the resolved elliptical-drop Application comparison runner with:
  `--warp-timestep-policy {pysph,current}` and `--warp-dt-max`.
- Implemented PySPH-like `n_damp` damping in the resolved runner and used
  output-time caps only for the current physical step.
- Updated experiment, aspect, daily, current, and session-log memory.
- Added resolved `nx=100` timestep-policy summary and side-by-side PNGs.

## Aspects touched and host files modified

- Aspects: `warp-backend`, `particle-memory`, `validation-benchmarks`,
  `host-integration`.
- Host files:
  - `pysph/base/warp_sph.py`
  - `pysph/base/tests/test_warp_sph.py`
- Experiment/memory files:
  - `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`
  - `.ai/implementations/blast-from-the-past/plans/2026-06-17_warp-adaptive-timestep-policy-parity.md`
  - `.ai/implementations/blast-from-the-past/current.md`
  - `.ai/implementations/blast-from-the-past/aspects/*/context.md`
  - `.ai/implementations/blast-from-the-past/updates/daily/2026-06-17.md`
  - `.ai/implementations/blast-from-the-past/updates/session-logs/2026-06-17_0824.md`

## Behavioral / numerical changes

- The resolved comparison can now run Warp with PySPH-like adaptive timestep
  policy: early `n_damp` sine ramp, adaptive growth beyond the initial
  timestep, and temporary output-time landing caps.
- Repeated particle state remains device-authoritative. The device still
  computes/reduces adaptive timestep factors; Python only receives and applies
  scalar timestep policy values.
- The old policy remains available as `--warp-timestep-policy current` for
  diagnostic comparisons.
- Full resolved `nx=100` result:
  - PySPH CPU Application: `233.97314716299297` s / 1393 steps.
  - Warp continuity-density: `23.629107111992198` s / 1393 steps.
  - Wall-time speedup: `9.901903870258701x`.
  - CPU `dt_max`: `2.780917055777183e-06`; Warp `dt_max`:
    `2.7813784981844947e-06`.
  - Warp `dt_min`: `2.7459356128852786e-09`, an output-time landing step.
- At `t=0.0038`, CPU-vs-Warp deltas were:
  major axis `-3.1258252297661215e-07`, minor axis
  `1.3598666296354978e-06`, `rho_min -8.228944999855159e-07`,
  `rho_max 9.947815438060559e-07`, kinetic energy
  `-0.0004318240680731833`.

## Validation

```text
python -m py_compile pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py
pass
```

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py -q -k 'adaptive_timestep_scale or adaptive_timestep_matches'
2 passed, 2 warnings
```

```text
python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
39 passed, 2 warnings in 4.83s
```

```text
python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 8 --output-times 0.0008 --prefix timestep-policy-smoke-nx8 --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/timestep-policy-smoke --max-steps 100000
PySPH CPU Application: 48 steps
Warp continuity-density: 48 steps
```

```text
python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-timestep-policy --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved --max-steps 10000000
PySPH CPU Application: 233.97314716299297 s, 1393 steps
Warp continuity-density: 23.629107111992198 s, 1393 steps
speedup_wall_time: 9.901903870258701
t=0.0008 major-axis delta: -5.596712788769054e-07
t=0.0038 major-axis delta: -3.1258252297661215e-07
t=0.0038 rho_min delta: -8.228944999855159e-07
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

- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-timestep-policy-t0p0008000.png`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved/resolved-nx100-timestep-policy-t0p0038000.png`

Visual inspection of the final checkpoint image shows CPU and Warp particle
clouds and exact ellipse overlays are visually indistinguishable at
`t=0.0038`.

## Risks / unresolved questions

- The timestep policy now mirrors the PySPH solver behavior closely enough for
  exact step-count parity in this case, but it is implemented in the resolved
  experiment runner, not as a general Solver integration.
- Adaptive timestep still transfers one scalar `dt` per step because launch
  parameters are host scalars.
- True periodic neighbor interactions remain out of scope for this slice.
- Raw HDF5/NPZ checkpoint dumps were generated locally and pruned from the
  reviewable artifact set; summary JSON and PNG comparisons remain.
- Unrelated untracked `CODEBASE_UNDERSTANDING.md` remains untouched.

## Reviewer verdict

- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-17T09:07:00 CEST
