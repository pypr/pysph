---
type: plan
id: 2026-06-16_warp-resolved-elliptical-drop-performance-comparison
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-16T23:09:00 CEST
status: approved
aspects: [validation-benchmarks, warp-backend, gpu-nnps, particle-memory, host-integration]
host_files: []
within_boundary: true
---

# Plan: Resolved Elliptical-Drop Performance and Results Comparison

## Goal

Run a resolved elliptical-drop case and compare Warp GPU results/performance
against PySPH CPU results at meaningful checkpoint times.

The target is not another tiny smoke. The target is a production-oriented
validation artifact with:

- matched initial condition and formulation;
- CPU PySPH baseline output;
- Warp GPU output;
- timing/per-step performance metrics;
- result metrics and side-by-side images.

## Baseline case

Use PySPH's elliptical-drop formulation as the reference:

- `Gaussian(dim=2)`;
- `rho0 = 1.0`;
- `c0 = 1400.0`;
- `gamma = 7.0`;
- `hdx = 1.3`;
- `alpha = 0.1`;
- `beta = 0.0`;
- XSPH correction enabled;
- adaptive timestep enabled with `cfl = 0.3`;
- canonical output times: `t = 0.0008` and `t = 0.0038`.

Resolution target:

- Default resolved case: `nx=100`.
- Use smaller cases only as debugging fallbacks if `nx=100` exposes a blocking
  runtime, stability, or output issue.
- Record any smaller fallback honestly as a fallback, not as the resolved
  benchmark.

## Approach

### Phase 1 - Baseline runner

- Add a resolved comparison runner/script under:
  `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`.
- Run PySPH CPU baseline using the existing no-scheme/Application path where
  practical so the output matches PySPH solver behavior.
- If full Application output is too slow for the first pass, run the existing
  CPU PySPH-primitive baseline and clearly label it as a fallback.

### Phase 2 - Warp resolved run

- Run the Warp elliptical-drop runner with:
  - Gaussian kernel / radius scale 3.0;
  - Tait EOS / per-particle `cs`;
  - artificial viscosity;
  - XSPH;
  - adaptive dt.
- Capture checkpoint arrays at the same target times as the CPU baseline.
- Keep repeated-step arrays device-authoritative; pull full particle arrays
  only at checkpoint/output times.

### Phase 3 - Performance metrics

- Record wall-clock runtime for CPU and Warp runs.
- Record:
  - total steps;
  - final simulated time;
  - min/max/mean dt;
  - particle count;
  - average step time;
  - checkpoint output time overhead if separately measurable.
- Include hardware/runtime summary from the active machine.

### Phase 4 - Result metrics and images

- Generate side-by-side images at each checkpoint:
  - CPU PySPH;
  - Warp GPU;
  - optional exact ellipse overlay from `pysph.examples.elliptical_drop.exact_solution`.
- Compute and record:
  - `x_min/x_max`, `y_min/y_max`;
  - major/minor axis estimates;
  - density min/max;
  - kinetic energy;
  - radius/shape summary;
  - particle count and all-finite checks.
- Summarize CPU-vs-Warp deltas for the above metrics.

## Files expected to change

- New resolved comparison script(s) under
  `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/`.
- New output images and small summary artifacts under the same experiment.
- The experiment markdown:
  `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- Memory/context:
  - `.ai/implementations/blast-from-the-past/current.md`
  - `.ai/implementations/blast-from-the-past/aspects/validation-benchmarks/context.md`
  - `.ai/implementations/blast-from-the-past/updates/daily/2026-06-16.md`
  - session log for this slice

No core host code changes are expected. If the resolved run exposes a core Warp
bug or missing capability, stop and write a new implementation plan before
changing `pysph/base/*`.

## Tests / validation

- Existing focused suite:
  `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- Resolved comparison script exits 0.
- CPU and Warp outputs exist at the target checkpoint times actually reached.
- Side-by-side images exist and are non-empty.
- Summary metrics report `all_finite == true` for CPU and Warp.
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai/implementations/blast-from-the-past`

## Success criteria

Minimum success:

- The default resolved case at `nx=100` runs CPU and Warp through at least
  `t = 0.0008`, unless a blocking issue is documented with a smaller fallback.
- CPU/Warp side-by-side image and metrics are recorded.
- Runtime and average step time are recorded for both.

Stretch success:

- `nx=100` reaches both `t = 0.0008` and `t = 0.0038`.
- Exact ellipse overlay and major/minor-axis deltas are recorded.
- Performance summary includes speedup and transfer/checkpoint overhead notes.

## Risks

- Full PySPH Application baseline may be slow at `nx=100`; the plan allows a
  smaller fallback only to diagnose blockers.
- `c0=1400` can make timestep very small. Runtime may dominate before physics
  issues appear.
- The current Warp path still has position wrapping only for periodic support,
  but elliptical drop does not require periodic boundaries.
- Any mismatch between the CPU PySPH Application integrator and the standalone
  Warp runner must be documented rather than hidden.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-16T23:14:00 CEST
- Approval, verbatim quote:
  > APPROVED
