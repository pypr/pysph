---
type: plan
id: 2026-07-06_warp-dynamic-adaptive-particle-resolution
author: @kunalpuri-prediqt
agent: codex
created: 2026-07-06T09:57:06 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files:
  - pysph/base/warp_adaptive.py
  - pysph/base/warp_device_helper.py
  - pysph/base/warp_nnps.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_adaptive.py
  - pysph/base/tests/test_warp_device_helper.py
  - pysph/base/tests/test_warp_nnps.py
  - pysph/base/tests/test_warp_sph.py
within_boundary: true
---

# Plan: Warp dynamic adaptive particle resolution

## Goal

Add single-GPU, device-resident dynamic particle refinement and coarsening to
the Warp WCSPH path so a 3D dam-break can use fine particles only in important
regions. Demonstrate the method on a fixed-obstacle dam-break with materially
fewer active particles and lower time-to-solution than a uniform simulation at
the same finest spacing, while preserving relevant physical observables.

This plan is deliberately bounded to the first credible adaptive production
path: fixed solid boundaries, geometry-driven resolution, a global timestep,
and one GPU. Local timestepping, solution-driven error estimation, moving rigid
bodies/contact, distributed memory, and multi-GPU execution are follow-ups.

## Context

The existing Warp backend already provides per-particle `m` and `h`, symmetric
variable-support neighbor inclusion, grid-direct fused WCSPH equations,
adaptive global timesteps, multi-array fixed solids, and a validated 3D
dam-break. These are necessary but not sufficient for adaptive resolution:

- `UniformGridWarpNNPS` chooses one cell size from the global largest `h` and
  reads coordinates/`h` to the host while recomputing bounds. A few coarse
  particles would therefore inflate neighbor work for every fine particle.
- `WarpDeviceHelper` can add, remove, and resize particles, but those operations
  currently use host NumPy copies/concatenation. Runtime APR must allocate,
  initialize, deactivate, and compact particles on the device.
- The WCSPH equations accept variable `m`/`h`, but the adaptive formulation
  still needs a selected 3D split/merge operator, variable-`h` consistency
  correction, property transfer, particle regularization, and free-surface/
  solid-boundary safeguards.
- The smallest active `h` controls the current global CFL step. APR can still
  save memory and neighbor work, but the fine region throttles all particles;
  this limitation must be measured honestly.

The primary algorithmic reference is Muta and Ramachandran, *Efficient and
Accurate Adaptive Resolution for Weakly-Compressible SPH* (2021), including
its open PySPH implementation (`https://gitlab.com/pypr/adaptive_sph`). Its
background resolution field, conservative split/merge workflow, neighbor-count
control, variable-`h` correction, and particle shifting are the starting point,
not a drop-in implementation: that work is 2D, EDAC-oriented, and does not
validate this 3D free-surface WCSPH dam-break. The 3D split/coalescence method
of Vacondio et al. (2016) and published GPU APR resource-management work are
additional evidence to evaluate before ADR-0007 selects the production design.

## Approach

### P0 - Freeze requirements, references, and non-adaptive baselines

1. Add reference notes for the Muta--Ramachandran PySPH implementation,
   Vacondio's conservative 3D splitting/coalescing method, and a GPU APR memory
   management implementation. Record which equations and dimensional claims
   are directly reusable and which require new validation.
2. Create an APR experiment packet with hardware, software versions, commands,
   exact initial conditions, and separate correctness/performance outputs.
3. Add the fixed-obstacle geometry to the Warp dam-break runner without APR:
   use the existing third solid array and `solid_indices=(1, 2)`. Validate this
   fixed-resolution case before using it to judge adaptivity.
4. Record uniform-fine and uniform-coarse baselines at sizes that fit the local
   GPU: particle count, peak device memory, neighbor candidates/accepted
   neighbors, step time, CFL history, mass, kinetic energy, surge front, maximum
   height, and obstacle pressure/impulse probes.
5. Run small 3D kill tests for candidate split patterns and merge selection.
   Compare mass/momentum conservation, density reconstruction error, neighbor
   count, isotropy, and GPU-friendly ownership. Do not choose a convenient
   eight-child cube without evidence.
6. Create ADR-0007 from the evidence. It must decide the split pattern, merge
   ownership rule, target-resolution representation, multilevel NNPS contract,
   adaptation cadence, boundary policy, and whether inactive-capacity storage
   or dense compaction is the canonical stepping representation.

Checkpoint: P0 has its own review and commit. No APR performance claim is made.

### P1 - Exact multilevel variable-resolution NNPS

1. Add a multilevel grid mode to `warp_nnps.py`. Quantize smoothing lengths to
   bounded resolution levels and build one device cell list per level with cell
   size proportional to that level's support radius.
2. Implement cross-level traversal that preserves the current symmetric
   inclusion rule: a pair is accepted when it falls inside either particle's
   support. The traversal must inspect every potentially overlapping source
   level without forcing fine particles through cells sized by global `hmax`.
3. Move bounds, per-level counts, offsets, scans, and grid metadata to the
   device. A small scalar readback may initially configure launches, but there
   must be no per-particle coordinate or `h` readback in the steady update.
4. Expose per-level diagnostics: active particles, occupied cells, candidate
   pairs, accepted neighbors, and rebuild time. Keep the existing uniform-grid
   mode unchanged for all current callers.
5. Integrate the new traversal into the generated equation-group launcher and
   adaptive-timestep neighbor reduction as a new explicit neighbor mode.

Checkpoint acceptance:

- Exact neighbor-set parity with brute force for randomized 3D distributions,
  empty levels, boundaries, and at least four levels spanning `h_max/h_min=16`.
- Fused equation outputs match the existing brute-force/grid oracle within
  dtype tolerance on mixed-`m`, mixed-`h` fixtures.
- Candidate work scales with local level populations; inserting one coarse
  particle does not globally resize fine cells.
- Existing uniform-grid and 2D generated-source tests remain unchanged.

### P2 - GPU particle pool, allocation, and compaction

1. Introduce `warp_adaptive.py` with a capacity-managed fluid particle pool.
   Track active slots, stable particle IDs, resolution level, target level, and
   all WCSPH/integrator properties required by split/merge and saved stages.
2. Add Warp kernels for flagging operations, exclusive-scan allocation,
   deterministic owner selection, child initialization, merge writes, active
   compaction, and old-to-new index maps. Use preallocated capacity and a
   documented growth factor; capacity overflow must fail cleanly or reallocate
   only at an explicit adaptation checkpoint.
3. Generalize `WarpDeviceHelper` only where needed to replace a complete device
   property set after compaction. Preserve its existing public behavior and do
   not rewrite unrelated host-compatible add/remove methods.
4. Rebuild NNPS level metadata and resize saved WCSPH state after compaction.
   No stale ParticleArray count, device wrapper, or saved-stage array may retain
   the pre-adaptation length.
5. Instrument host transfers and allocations. Routine adaptation after warm-up
   must not pull particle properties to NumPy or allocate one device array per
   new child.

Checkpoint acceptance:

- Repeated synthetic refine/coarsen cycles have no lost, duplicate, or
  multiply-owned slots and preserve stable IDs for unchanged particles.
- All strided/scalar properties compact consistently; zero/empty/full-capacity
  and overflow paths are tested.
- Mass and linear momentum are conserved by the raw device operations to
  dtype-derived tolerance (`<=1e-12` relative in f64 and `<=1e-6` in f32 for
  well-scaled fixtures). Any non-conserved quantity is explicitly documented.
- Steady adaptation performs no per-particle host transfer.

### P3 - Conservative 3D split/merge and variable-resolution WCSPH

1. Implement the ADR-selected 3D split operator. Initialize daughter position,
   mass, smoothing length, velocity, density, pressure, sound speed, and
   integrator state from the parent/local reconstruction.
2. Implement deterministic parallel merge candidate selection and conservative
   coalescence. Preserve total mass and linear momentum; minimize density error
   and quantify angular-momentum and kinetic-energy changes rather than hiding
   them.
3. Add the selected variable-`h`/partition-consistency correction to sibling APR
   equation blocks. Do not modify existing uniform WCSPH generated blocks.
4. Add iterative particle shifting/regularization and first-order property
   correction. Prevent shifting across the free surface or through fixed solids.
5. Recompute `h` from local target mass/neighbor population and restrict jumps
   between adjacent resolution regions according to ADR-0007.
6. Adapt only every `n_adapt` steps (configurable). The default cadence is
   selected from measured adaptation cost and transient error, not assumed to
   be every step.

Checkpoint acceptance:

- Constant fields are reproduced through split/merge/shift to roundoff; linear
  field and kernel-summation errors have recorded convergence behavior.
- Hydrostatic and translating-fluid fixtures survive repeated crossings of a
  refinement boundary without secular density/pressure growth.
- Conservation and density error meet the P2 gates after complete property
  reconstruction, not only after raw slot operations.
- A uniform target level routes through the APR sibling path and matches the
  existing uniform dam-break step at fp32/fp64 tolerance.

### P4 - Geometry-driven adaptive obstacle dam-break

1. Add a background target-resolution field for the flume. The first policy is
   geometry driven and deterministic: finest near the obstacle, impact/probe
   region, and selected free-surface band; progressively coarser in quiet bulk
   fluid. Fluid particles split or merge as they move through this field.
2. Keep fixed wall/obstacle particles static and generate their local spacing
   once so neighboring fluid/solid resolution ratios remain within the
   ADR-selected limit. Dynamic boundary-particle adaptation is deferred.
3. Add a sibling adaptive dam-break driver. The existing fixed-wall and
   rigid-body drivers remain unchanged.
4. Validate first on two levels, then raise the number of levels only after the
   transition diagnostics pass. Save level-colored scientific snapshots and
   adaptation histories; do not infer correctness from a plausible animation.
5. Compare against a uniform-fine run at the same finest spacing wherever it
   fits. For a larger demonstration that cannot fit uniformly, report the
   uniform-equivalent particle/memory estimate separately from measured active
   particle count and measured peak memory.

Final scientific gates:

- All state remains finite and device error is zero.
- Relative total-mass drift attributable to adaptation is `<=1e-5` over the
  benchmark; adaptation-event momentum residual is `<=1e-5` relative.
- Surge-front position and maximum height remain within two finest-particle
  spacings of the uniform-fine reference at matched checkpoints.
- Obstacle pressure impulse and developed-flow kinetic energy remain within 5%
  of the uniform-fine reference. If fp32 chaos makes pointwise peak pressure
  unsuitable, use time-windowed impulse and disclose that choice in the review.
- Resolution-transition density error and neighbor-count distributions remain
  bounded with no persistent void/banding at level interfaces.

Final scale gates:

- At least 4x fewer peak active particles than the equivalent uniform-fine
  discretization for the selected case.
- At least 2x lower measured time-to-solution than a uniform-fine case that fits
  on the same GPU. If the global fine-particle timestep prevents this, report
  the miss and do not claim scale success.
- One adaptive case whose equivalent uniform-fine state would exceed available
  device memory completes within measured device capacity. Label the uniform
  figure as an estimate and keep correctness anchored to the smaller runnable
  uniform reference.

Each phase receives a review artifact and exact `@prabhu: LGTM` before its
commit. Later phases may amend numerical thresholds through reviewed evidence,
but may not silently weaken them.

## Files expected to change

- `pysph/base/warp_adaptive.py` (new) - device particle pool, target-resolution
  data, split/merge/shift kernels, conservation diagnostics, and adaptation
  orchestration.
- `pysph/base/warp_nnps.py` - additive multilevel grid and cross-level search;
  existing uniform mode retained.
- `pysph/base/warp_device_helper.py` - minimal device-property replacement/
  compaction integration.
- `pysph/base/warp_sph.py` - additive APR equation blocks, multilevel launcher
  routing, variable-`h` correction, adaptive timestep routing, and sibling
  adaptive dam-break driver.
- `pysph/base/tests/test_warp_adaptive.py` (new) - particle-pool, conservation,
  reconstruction, shifting, and repeated adaptation tests.
- Existing focused Warp test files listed in frontmatter - regression, NNPS
  parity, helper lifetime, generated-source, and solver integration tests.
- `.ai/implementations/blast-from-the-past/experiments/` - new APR kill tests,
  obstacle runner, uniform/adaptive outputs, performance data, and figures.
- ADR-0007, reference notes, aspect contexts/issues, reviews, session logs,
  closeouts, and `current.md` as required by the operating contract.

All planned host files are within the existing `pysph/base/warp_*.py` and
`pysph/base/tests/test_warp_*.py` integration boundary. If implementation shows
that changes to generic ParticleArray, Cython ABI, solver, or shipped examples
are required, stop and amend this plan with `within_boundary: false` before
touching them.

## Tests / validation

- Focused unit tests after every phase using the active project Python.
- Brute-force CPU/NumPy oracles for neighbor sets, split/merge conservation,
  interpolation, density, and one-step APR behavior.
- Existing full `test_warp_*` suite and the complete Warp SPH suite at each
  reviewed checkpoint; final `python -m pytest -m "not slow" pysph` if runtime
  is practical, otherwise record the focused/full split and reason.
- Fixed-obstacle uniform CPU-vs-Warp comparison before APR validation.
- Repeated performance runs after warm JIT/cache, including device model,
  driver, Warp version, precision, commit, commands, active/equivalent particle
  counts, memory, adaptation cost, neighbor work, step count, and physics deltas.
- `validate-memory.py`, decision-graph regeneration when ADR-0007 lands,
  `git diff --check`, and adversarial review before every commit.

## Risks

- The published PySPH adaptive method is not a validated 3D free-surface WCSPH
  recipe. Porting mechanics without re-deriving the formulation could conserve
  mass yet produce incorrect pressure and density at transitions.
- In 3D, each factor-of-two spacing refinement changes local particle count by
  roughly eight. Poor target-field hysteresis can cause explosive particle
  growth or split/merge thrashing.
- Cross-level NNPS omissions are silent physics errors. Brute-force set parity
  is a hard gate before any performance work.
- Particle shifting can move particles through a free surface or solid unless
  classification and displacement limits are correct.
- A global timestep is controlled by the finest particles. Memory savings may
  exceed runtime savings; local/multirate stepping is intentionally not hidden
  inside this first plan.
- Device compaction changes particle indices and can invalidate saved state,
  rigid mappings, output identity, or cached grids. Stable IDs and explicit
  old/new maps are required.
- Warp JIT source size and compile time are already high for 3D Wendland groups.
  New APR groups must remain additive and be compiled/tested incrementally.

## Out of scope

- Local, asynchronous, or multirate timestepping.
- Multi-GPU/MPI domain decomposition, migration, and load balancing.
- Moving rigid bodies, rigid contact, and dynamically adapted solid boundaries.
- Solution/error-estimator-driven refinement (vorticity, pressure-gradient,
  learned indicators) beyond recording the extension points.
- Arbitrary continuous `h`; the first production path uses bounded levels.
- Replacing existing uniform Warp paths or changing public Cython ABI/API.
- Photorealistic rendering.

## Estimated effort

Five reviewed implementation checkpoints. This is a multi-session research and
engineering track, expected to exceed 1,500 LOC across backend, tests, and
experiment runners. P0 and P1 are kill gates: if exact multilevel neighbor
search or variable-resolution physics cannot meet their correctness gates, the
plan stops before device-pool complexity or scale claims accumulate.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-07-06T10:34:32 CEST
- Approval, verbatim quote:
  > approved
