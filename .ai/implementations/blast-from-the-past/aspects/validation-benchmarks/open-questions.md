# Open Questions - validation-benchmarks

- [open] What does "blazing fast" mean in concrete speedup, throughput, and hardware terms?
- [open] What correctness and timing thresholds should promote elliptical drop
  from smoke workload to first published particle-dynamics benchmark?
- [closed 2026-06-17] For apples-to-apples resolved Application comparisons,
  the Warp runner should use PySPH-like adaptive timestep policy: `n_damp`
  growth and temporary output-time landing caps. The old initial-`dt` capped
  policy remains available as `--warp-timestep-policy current` for diagnostics.
