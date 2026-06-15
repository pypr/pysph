# Current - blast-from-the-past

Updated: 2026-06-15T12:30:00 CET by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, uniform-grid/cell-list baselines, a device-resident neighbor-sum consumer, and a real Warp `SummationDensity` kernel. `warp_grid_density` shows `69.084x` CPU/Cython speed at 10,000,000 particles.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`; `experiments/2026-06-15_warp-nnps-device-consumption`; `experiments/2026-06-15_warp-summation-density`.
**Open approvals:** None for the current brute-force NNPS baseline; broader Application integration still needs a decision.
**Next action:** Generalize the one-equation `compute_summation_density()` proof into a reusable Warp equation-loop contract.
