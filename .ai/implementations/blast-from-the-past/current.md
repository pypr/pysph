# Current - blast-from-the-past

Updated: 2026-06-15T11:20:00 CET by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, uniform-grid/cell-list baselines, and a device-resident neighbor-sum consumer, with `warp_grid_reduce` showing `145.583x` CPU speed at 1,000,000 particles.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`; `experiments/2026-06-15_warp-nnps-device-consumption`.
**Open approvals:** None for the current brute-force NNPS baseline; broader Application integration still needs a decision.
**Next action:** Turn the one-off `compute_neighbor_sum()` proof into a reusable Warp equation-loop contract.
