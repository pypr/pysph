# Current - blast-from-the-past

Updated: 2026-06-15T23:27:00 CET by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, uniform-grid/cell-list baselines, a device-resident neighbor-sum consumer, and real Warp SPH kernels for `SummationDensity`, `IsothermalEOS`, `ContinuityEquation`, inviscid pressure-gradient acceleration, and a minimal one-step WCSPH Euler update. `warp_grid_pgrad` shows `38.722x` CPU/Cython speed at the capped 5,000,000-particle pgrad benchmark, and the one-step WCSPH chain matches CPU reference state values in focused tests.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`; `experiments/2026-06-15_warp-nnps-device-consumption`; `experiments/2026-06-15_warp-summation-density`; `experiments/2026-06-15_warp-eos-continuity`; `experiments/2026-06-15_warp-pressure-gradient`; `experiments/2026-06-15_warp-wcsph-euler-step`.
**Open approvals:** None for the current brute-force NNPS baseline; broader Application integration still needs a decision.
**Next action:** Add a repeated-step loop with a device-aware NNPS refresh after position updates; avoid treating stale host ParticleArray coordinates as the source of truth between GPU steps.
