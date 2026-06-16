---
type: review
date: 2026-06-16
user: @kunalpuri-prediqt
agent: codex
plan: session-log LP-N
adrs: []
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: [pysph/base/warp_nnps.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_nnps.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Warp repeated-step leapfrog checkpoint

## Diff summary

- Adds `update(push=False)` to Warp NNPS update paths so grids/caches can be
  rebuilt from device-updated coordinates without pushing stale host
  `ParticleArray` values over them.
- Adds Warp KDK leapfrog primitives: half/full kick, drift, periodic position
  wrapping, and a minimal `wc_sph_leapfrog_step()`.
- Adds `push=False` controls to Warp summation-density and continuity helpers
  for device-authoritative step loops.
- Adds correctness tests for device-coordinate NNPS refresh, direct
  kick/drift/wrap behavior, and KDK WCSPH state against CPU references.
- Updates implementation memory and experiment expectations/results.

## Aspects touched and host files modified

- `warp-backend`: new Warp kernels and helper functions in
  `pysph/base/warp_sph.py`.
- `gpu-nnps`: device-authoritative refresh in `pysph/base/warp_nnps.py`.
- `particle-memory`: avoids stale host-to-device pushes in repeated step loops.
- `validation-benchmarks`: focused tests and experiment doc updated.
- `host-integration`: no Application integration yet; boundary drift noted
  below.

Host files:

- `pysph/base/warp_nnps.py`
- `pysph/base/warp_sph.py`
- `pysph/base/tests/test_warp_nnps.py`
- `pysph/base/tests/test_warp_sph.py`

## Behavioral / numerical changes

- Existing default `update()` behavior remains host-authoritative.
- New `update(push=False)` mode allows the Warp device arrays to be the source
  of truth after device-side position updates.
- `wc_sph_leapfrog_step()` computes:

```text
a_n     <- WCSPH acceleration(x_n)
u_half  <- u_n + 0.5*dt*a_n
x_np1   <- x_n + dt*u_half
wrap x_np1 into periodic bounds when requested
NNPS refresh from device x_np1
a_np1   <- WCSPH acceleration(x_np1)
u_np1   <- u_half + 0.5*dt*a_np1
```

- Periodic support in this change is position wrapping only. Minimum-image
  neighbor distances and periodic cell lookup are not implemented here.

## Tests / validation run

```text
$ python -m pytest -q pysph/base/tests/test_warp_sph.py pysph/base/tests/test_warp_nnps.py
.............................                                            [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
29 passed, 2 warnings in 5.82s
```

```text
$ bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_warp-wcsph-euler-step/run_correctness.sh
.............................                                            [100%]
=============================== warnings summary ===============================
pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:29: DeprecationWarning: Due to '_pack_', the 'APICLaunchParamRecord' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchParamRecord(ctypes.Structure):

pysph/base/tests/test_warp_sph.py::test_warp_isothermal_eos_matches_cpu_and_pulls_pressure
  /home/kunalp/.pqt_venv_e0b41259/lib/python3.14/site-packages/warp/_src/apic/types.py:49: DeprecationWarning: Due to '_pack_', the 'APICLaunchPtrLocation' Structure will use memory layout compatible with MSVC (Windows). If this is intended, set _layout_ to 'ms'. The implicit default is deprecated and slated to become an error in Python 3.19.
    class APICLaunchPtrLocation(ctypes.Structure):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
29 passed, 2 warnings in 3.05s
```

```text
$ git diff --check -- pysph/base/warp_nnps.py pysph/base/warp_sph.py pysph/base/tests/test_warp_nnps.py pysph/base/tests/test_warp_sph.py .ai/implementations/blast-from-the-past
<no output>
```

## validate-memory.py

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: yes
- Amendments log entry: `2026-06-16 - Added Python Warp prototype files and
  focused Warp tests to the active implementation boundary for the repeated-step
  checkpoint.`
- Note: plan `2026-06-16_warp-repeated-step-leapfrog-and-periodic-refresh.md`
  was marked `within_boundary: false`; the boundary was amended during review
  before commit.

## Visual aid

Comparison table, Mermaid diagram, generated chart, convergence plot, or one-line waiver.

| Step | Before | After |
| --- | --- | --- |
| NNPS refresh after device drift | `update()` pushed host coordinates first | `update(push=False)` rebuilds from device coordinates |
| Integrator prototype | Euler only | Euler plus KDK leapfrog |
| Periodic support | none in step helper | position wrapping after drift |
| Validation | 26 focused tests | 29 focused tests plus wrapper run |

## Risks

- True periodic neighbor interaction is still incomplete: position wrapping is
  not enough without minimum-image distances and periodic cell lookup.
- `wc_sph_leapfrog_step()` is still a prototype helper, not a generated PySPH
  integrator or Application-level solver path.
- The remaining neighbor-cache sizing path still reads lengths to host.
- The Python Warp prototype files need a boundary amendment decision.

## Unresolved questions

- Should the next milestone be true periodic Warp NNPS, artificial viscosity, or
  an Application-facing elliptical-drop runner?
- What exact acceptance threshold should gate an elliptical-drop comparison
  against PySPH's analytical/post-process output?

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM - 2026-06-16T12:26:17 CEST
