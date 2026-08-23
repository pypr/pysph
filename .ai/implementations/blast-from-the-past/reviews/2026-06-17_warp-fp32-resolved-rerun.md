---
type: review
date: 2026-06-17
user: @kunalpuri-prediqt
agent: codex
plan: .ai/implementations/blast-from-the-past/updates/session-logs/2026-06-17_0930.md
adrs: []
aspects_touched: [particle-memory, validation-benchmarks, warp-backend]
host_files: []
status: lgtm
---

# Review: Warp fp32 Resolved Rerun

## Diff summary

- Recorded an explicit Warp-only fp32 `nx=100` resolved elliptical-drop rerun.
- Added the compact summary JSON under the elliptical-drop experiment's
  `fp32/` output directory.
- Updated experiment notes to clarify current precision behavior:
  `WarpEllipticalDropRunner` builds host arrays as `float64`, but
  `WarpDeviceHelper` casts floating device properties according to
  `compyle.config.get_config().use_double`; on this machine it is `False`, so
  the current Warp device execution path is fp32.
- Updated validation-benchmark context, current pointer, daily closeout, and a
  session log for the fp32 precision check.

## Behavioral / numerical changes

- No host code changed.
- No runner behavior changed.
- The explicit fp32 Warp-only rerun used:
  - `nx=100`
  - `c0=1400.0`
  - `density_mode=continuity`
  - `warp_timestep_policy=pysph`
  - checkpoints `0.0008,0.0038`
- Result:
  - wall time `26.5732471299998` s
  - steps `1393`
  - average step time `0.019076272167982626` s
  - `all_finite=true` at both checkpoints
  - CPU-baseline-relative speedup, using the committed CPU time
    `233.97314716299297` s: `8.804838415808415x`
- The previous committed Warp timing was `23.629107111992198` s / 1393 steps;
  both runs use the same fp32 device path, so the difference is treated as
  run-to-run/module-cache variance rather than a precision effect.

## Validation

```text
python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-fp32-warp-only --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/fp32 --skip-pysph-application --max-steps 10000000
Warp fp32 wall time: 26.5732471299998 s
steps: 1393
average step time: 0.019076272167982626 s
all_finite: true at both checkpoints
```

```text
python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

```text
git diff --check -- .ai/implementations/blast-from-the-past
pass
```

## Visual aid

Waived. This was a Warp-only timing/precision rerun with
`--skip-pysph-application`, so no side-by-side plot was generated. The retained
artifact is the compact summary JSON:

```text
.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/fp32/resolved-nx100-fp32-warp-only-summary.json
```

## Risks / unresolved questions

- A true fp64 Warp comparison still needs either setting
  `compyle.config.get_config().use_double = True` before creating the Warp
  ParticleArray or adding an explicit runner precision flag.
- The raw fp32 checkpoint `.npz` files were pruned; only the summary JSON is
  retained.
- Unrelated untracked `CODEBASE_UNDERSTANDING.md` remains untouched.

## Reviewer verdict

- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-17T09:43:00 CEST
