# Current - blast-from-the-past

Updated: 2026-06-16T12:15:00 CEST by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, uniform-grid/cell-list baselines, a device-resident neighbor-sum consumer, and real Warp SPH kernels for `SummationDensity`, `IsothermalEOS`, `ContinuityEquation`, inviscid pressure-gradient acceleration, one-step WCSPH Euler, and a minimal KDK leapfrog step. `UniformGridWarpNNPS.update(push=False)` can rebuild from device-updated coordinates, leapfrog kick/drift and periodic position wrapping run on device, and the focused Warp SPH/NNPS suite passes with `29 passed`.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`; `experiments/2026-06-15_warp-nnps-device-consumption`; `experiments/2026-06-15_warp-summation-density`; `experiments/2026-06-15_warp-eos-continuity`; `experiments/2026-06-15_warp-pressure-gradient`; `experiments/2026-06-15_warp-wcsph-euler-step`.
**Open approvals:** Plan `plans/2026-06-16_warp-repeated-step-leapfrog-and-periodic-refresh.md` was approved and implemented locally; broader Application integration still needs a decision.
**Next action:** Add true periodic neighbor interactions through minimum-image distance/cell lookup, then add the next WCSPH force term such as artificial viscosity before broader solver/Application integration.
