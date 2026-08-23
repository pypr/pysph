---
type: review
date: 2026-06-17
user: @kunalpuri-prediqt
agent: codex
aspects_touched: [validation-benchmarks, warp-backend, gpu-nnps]
host_files: []
plan: none
adrs: []
status: approved
verdict: "@prabhu: LGTM"
---

# Review - Million-Particle Short Comparison

## Diff summary

- Recorded a bounded million-particle, ten-step CPU/GPU elliptical-drop
  comparison in the active Warp elliptical-drop experiment.
- Added compact JSON summary evidence for the run.
- Updated `current.md`, today's daily closeout, and a new session log.
- Recorded that the full million-particle GPU-only run was not launched after
  the user clarified "no multi hour run".

## Aspects

- `validation-benchmarks`
- `warp-backend`
- `gpu-nnps`

## Host files

None. This review covers experiment/memory artifacts only.

## Behavioral / numerical changes

No host behavior changed.

The new recorded benchmark result is:

```text
nx=565
particles=1,002,885
steps=10
fixed dt=3.732778967800475e-07
PySPH CPU Application wall time: 57.48 s
Warp GPU wall time: 7.17 s
speedup_wall_time: 8.01673640167364x
```

Final-state deltas:

```text
axis_x_abs: -1.2296967044633789e-07
axis_y_abs: 4.773760275966765e-10
rho_min: -9.119009991565008e-10
rho_max: 1.4501548406542497e-08
kinetic_energy: 8.523681572114583e-06
```

The one-step adaptive GPU probe completed in `4.87 s` and stayed finite, but
the full `nx=565`, `tf=0.0076` GPU-only run was not launched.

## Raw validation output

```text
python pysph/examples/elliptical_drop_no_scheme.py --nx 565 --tf 0.000003732778967800475 --timestep 0.0000003732778967800475 --no-adaptive-timestep --n-damp 0 --pfreq 10 --fname million-pysph --directory .../million-cpu-gpu-10step/pysph --logfile '' --quiet
real 57.48
```

```text
python .../warp_elliptical_drop_runner.py --nx 565 --steps 10 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --density-mode continuity --output .../million-cpu-gpu-10step/warp/million-warp.npz
real 7.17
```

```text
python .../warp_elliptical_drop_runner.py --nx 565 --steps 1 --dt 0.0000003732778967800475 --rho0 1.0 --c0 1400.0 --p0 0.0 --alpha 0.1 --beta 0.0 --eos tait --gamma 7.0 --kernel gaussian --xsph-eps 0.5 --adaptive-dt --cfl 0.3 --dt-min 1.0e-10 --dt-max 0.0000003732778967800475 --density-mode continuity --output .../full-nx565-gpu/adaptive-one-step-probe.npz
real 4.87
all_finite: true
```

```text
python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

```text
ps -ef | rg 'resolved_elliptical_drop_comparison|warp_elliptical_drop_runner|elliptical_drop_no_scheme.py' | rg -v rg
<no output>
```

## Risks

- The committed evidence intentionally excludes the large raw HDF5/NPZ outputs.
  The compact summary JSON is the committed benchmark evidence.
- The ten-step run is fixed-step and short by design. It is not a substitute
  for a full-duration resolved million-particle physics result.
- The full-duration million-particle adaptive path likely needs optimization
  before a non-multi-hour run is practical.

## Unresolved questions

- What optimization target should gate a full `nx=565`, `tf=0.0076` GPU-only
  elliptical-drop run?
- Should future large-output artifacts be archived outside git with checksums?

## Visual aid

Waiver: no new plot was generated for this review because plotting 1,002,885
particles would add bulky derived artifacts. The numerical summary is the
review evidence for this experiment checkpoint.

## Verdict

Approved by @prabhu with verbatim quote:

```text
@prabhu: LGTM
```
