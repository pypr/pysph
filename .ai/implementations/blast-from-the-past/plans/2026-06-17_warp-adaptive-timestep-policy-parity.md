---
type: plan
id: 2026-06-17_warp-adaptive-timestep-policy-parity
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-17T08:24:00 CEST
status: approved
aspects: [warp-backend, particle-memory, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp Adaptive Timestep Policy Parity

## Goal

Match PySPH's adaptive timestep policy closely enough that the resolved
elliptical-drop CPU/GPU step counts are comparable under the same timestep
controller.

The continuity-density comparison already fixed the physics/formulation
mismatch. The remaining step-count difference is now mostly policy:

- PySPH CPU: 1393 steps, `dt_mean ~= 2.740e-6`, `dt_max ~= 2.781e-6`;
- Warp GPU: 1804 steps, `dt_mean ~= 2.106e-6`, `dt_max ~= 2.109e-6`.

The Warp resolved runner currently caps `dt_max` at the initial timestep, while
PySPH's adaptive solver allows the timestep to grow after applying the
`n_damp` ramp.

## PySPH behavior to mirror

From `pysph/solver/solver.py` and `pysph/sph/integrator.py`:

- The integrator computes a local adaptive timestep as
  `cfl * min(hmin/dt_cfl_fac, sqrt(hmin/sqrt(dt_force_fac)), hmin/dt_visc_fac)`.
- The solver starts from `_get_undamped_timestep()`, where
  `_get_undamped_timestep()` is `self.dt / self._damping_factor`.
- For early steps, `n_damp` applies:
  `0.5 * (sin(pi * (-0.5 + (count + 1)/n_damp)) + 1.0)`.
- Output-time handling can temporarily shorten `self.dt`, but PySPH preserves
  `_prev_dt` and restores it after the output step. Solver output records
  undamped `dt`, not necessarily the shortened checkpoint step.

## Approach

### Phase 1 - Focused timestep-policy helper

- Add a small Python-side helper for the experiment runner that mirrors PySPH's
  solver timestep policy:
  - track `count`;
  - track damping factor;
  - compute the undamped adaptive candidate from the device-reduced Warp
    timestep helper;
  - apply the `n_damp` sine ramp;
  - cap only for output-time landing as a temporary step, then restore the
    previous undamped candidate for the next step.
- Keep the device work unchanged: adaptive factors remain computed/reduced on
  device, with only scalar `dt` copied to host.

### Phase 2 - Warp helper interface

- If needed, expose a lower-level Warp adaptive timestep candidate that returns
  the undamped device-reduced value before runner-level damping/output-time
  policy.
- Preserve current `compute_wcsph_adaptive_timestep()` behavior for existing
  tests/callers unless a compatibility-preserving option is clearer.
- Avoid full host particle-array pulls/pushes inside the repeated stepping loop.

### Phase 3 - Runner and resolved comparison

- Update `resolved_elliptical_drop_comparison.py` to use the PySPH-like policy
  when `--warp-density-mode continuity` is used.
- Add CLI flags if useful:
  - `--warp-timestep-policy {current,pysph}`;
  - default the resolved comparison to `pysph` policy;
  - preserve the old behavior for diagnostics.
- Record both the actual shortened step history and PySPH-style restored
  reported timestep history if needed to interpret output.

### Phase 4 - Validation

- Add focused tests for the damping factor and output-time restore behavior.
- Run the focused Warp suite:
  `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- Run a small `nx=8` resolved comparison to check wiring.
- Run the resolved `nx=100` Application comparison through `t=0.0008` and
  `t=0.0038`.
- Update the experiment report with:
  - CPU/Warp step counts;
  - `dt_min`, `dt_mean`, `dt_max`;
  - shape/density/energy deltas;
  - whether step counts now match or what remaining difference remains.

## Files expected to change

- `pysph/base/warp_sph.py` only if a lower-level adaptive candidate helper or
  option is needed.
- `pysph/base/tests/test_warp_sph.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py`
- `.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/experiment.md`
- New/updated lightweight resolved summary/images under the same experiment.
- Memory/context:
  - `.ai/implementations/blast-from-the-past/current.md`
  - `.ai/implementations/blast-from-the-past/aspects/validation-benchmarks/context.md`
  - `.ai/implementations/blast-from-the-past/aspects/validation-benchmarks/open-questions.md`
  - `.ai/implementations/blast-from-the-past/updates/daily/2026-06-17.md`
  - session log for this slice

## Tests / validation

- `python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py`
- `python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 8 --output-times 0.0008 --prefix timestep-policy-smoke-nx8 --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/timestep-policy-smoke --max-steps 100000`
- `python .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved_elliptical_drop_comparison.py --nx 100 --output-times 0.0008,0.0038 --prefix resolved-nx100-timestep-policy --output-dir .ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner/resolved --max-steps 10000000`
- `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
- `git diff --check -- .ai/implementations/blast-from-the-past pysph/base/warp_sph.py pysph/base/tests/test_warp_sph.py`

## Success criteria

- Warp no longer hard-caps the resolved run to the initial timestep when using
  the PySPH-like policy.
- `n_damp=50` changes the early Warp timestep history in the same way PySPH
  does.
- Output-time landing does not permanently shrink the next Warp timestep.
- The resolved `nx=100` comparison remains finite and visually/metric-wise
  close to PySPH CPU.
- Step-count differences are either eliminated or reduced and explained with a
  concrete remaining policy/integrator reason.

## Risks

- PySPH records/restores timestep values around output differently from the
  raw physical step actually taken, so we may need to store both histories to
  explain comparisons honestly.
- Exact step-count parity may still differ if PySPH's acceleration evaluation
  timing updates `dt_cfl`/`dt_force` at a subtly different point in the PEC
  cycle.
- The full `nx=100` Application rerun takes several minutes.

## Approval

- [ ] Plan posted in chat
- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-17T08:25:00 CEST
- Approval, verbatim quote:
  > approved
