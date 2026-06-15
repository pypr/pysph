# Current - blast-from-the-past

Updated: 2026-06-15T13:25:00 CET by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, uniform-grid/cell-list baselines, a device-resident neighbor-sum consumer, and real Warp SPH kernels for `SummationDensity`, `IsothermalEOS`, and `ContinuityEquation`. `warp_grid_eos_cont` shows `72.583x` CPU/Cython speed at the capped 5,000,000-particle EOS+continuity benchmark.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`; `experiments/2026-06-15_warp-nnps-device-consumption`; `experiments/2026-06-15_warp-summation-density`; `experiments/2026-06-15_warp-eos-continuity`.
**Open approvals:** None for the current brute-force NNPS baseline; broader Application integration still needs a decision.
**Next action:** Add pressure-gradient momentum, then generalize the one-off Warp SPH kernels into a reusable equation-loop contract.
