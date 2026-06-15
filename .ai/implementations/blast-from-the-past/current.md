# Current - blast-from-the-past

Updated: 2026-06-15T10:10:00 CET by codex

**Status:** Initial Warp ParticleArray device mirror prototype is committed locally; codebase understanding is folded into the spec tree; Warp NNPS now has brute-force, cached-flat-list, and uniform-grid/cell-list baselines, with `warp_grid_device` showing `88.288x` CPU speed at 1,000,000 particles.
**Active aspects:** warp-backend, gpu-nnps, particle-memory, cython-boundary, validation-benchmarks, host-integration
**In-flight experiments:** `experiments/2026-06-15_initial-warp-benchmark-placeholder`; `experiments/2026-06-15_warp-nnps-bruteforce-baseline`.
**Open approvals:** None for the current brute-force NNPS baseline; broader Application integration still needs a decision.
**Next action:** Decide how equation kernels should consume the device-resident Warp grid neighbor cache.
