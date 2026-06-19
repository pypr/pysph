---
aspect: warp-backend
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-18T13:30:00 CEST
status: active
---

# Aspect: warp-backend

## What this aspect covers

NVIDIA Warp API choices, kernel model, memory layout assumptions, and how Warp could map onto PySPH's existing GPU abstractions.

## Current understanding

The first likely Warp integration boundary is a DeviceHelper-like mirror, not a replacement of ParticleArray host storage. The spec identifies required Warp primitives: device array creation, selective/full push and pull, Local-first partition/alignment, strided gather/scatter, resize/fill, add/remove/extract/append, and min/max if parity with current helper is desired.

Warp imports successfully in the active environment as version `1.14.0`. Before code, decide whether Warp appears as a new `backend='warp'`, a CUDA backend variant, or a separate helper.

ADR-0002 accepted the DeviceHelper-like mirror direction. The prototype adds `pysph/base/warp_device_helper.py`, exposes `backend='warp'` through ParticleArray backend resolution, and uses Warp gather kernels for alignment over scalar and strided properties. It now also supports remove, remove-tagged, add, append, extend, and extract through ParticleArray public methods. Focused Warp helper tests cover the main prototype surface and pass against a rebuilt `pysph.base.particle_array` extension.

Warp equation kernels now cover the current elliptical-drop smoke formulation:
CubicSpline/Gaussian kernel selection, summation density, isothermal/Tait EOS,
continuity, pressure-gradient acceleration, Monaghan artificial viscosity,
XSPH correction, KDK leapfrog, WCSPH PEC-style continuity-density staging,
periodic position wrapping, and device-reduced WCSPH adaptive timestep factors.
The leapfrog helper exposes scalar policy controls for adaptive timesteps:
`adaptive_dt_scale` applies PySPH-style damping after the device reduction and
`step_dt_max` caps only the current physical step, leaving full particle state
device-resident.

## Key sub-topics

- Warp version/API surface - Active environment has Warp `1.14.0`; confirm documentation set with team.
- Kernel launch model for partition/gather/scatter kernels - Confirm with prototype.
- Compatibility with existing PySPH GPU pathways.
- Backend naming and ownership ADR.
- Next kernel family decision: move add/remove/extract/append growth internals from host-side NumPy concatenation to fully device-side Warp kernels.
- Tutorial documentation added at `docs/source/tutorial/warp_particle_array.rst`.
- Adaptive timestep reductions currently transfer only the final scalar `dt`
  back to Python because launch parameters remain host scalars. Runner-level
  damping and checkpoint caps are scalar-only policy operations.
- The PySPH Application parity step uses device-side saved state plus
  continuity-density PEC stages; the original summation-density KDK path remains
  the compatibility default.
- ADR-0003 adopts dynamic Warp equation-group code generation
  (`pysph/base/warp_codegen.py`), mirroring PySPH's equation/group transpilation
  on the GPU. A `WarpEquation` block declares its source/dest/out arrays, the
  shared per-pair quantities it needs (`dx`, `rij`, `hij`, `grad`, `wij`,
  `vij*`), and `initialize`/`loop`/`post_loop` source snippets; a group unions
  the signature, computes shared geometry once, inlines each block's per-pair
  loop into one neighbor traversal accumulating into shared `_acc_<out>`
  registers, and emits one cached, JIT-compiled kernel per
  `(ordered equation signatures, dtype)`. Kernels are materialized by templating
  source, registering it in `linecache`, and wrapping with
  `wp.Kernel(func=..., source=...)`; the generated namespace is seeded with the
  module's device `wp.func`s so Warp resolves them. Fusion is thus a property of
  grouping rather than hand-written, and the f32/f64 split collapses into a
  dtype parameter.
- First consumer: the continuity-density PEC half-stage fuses pressure gradient,
  Monaghan viscosity, continuity, and XSPH (blocks in
  `_WCSPH_CONTINUITY_BLOCKS`) into one generated kernel via
  `compute_wcsph_accel_continuity`. The summation-density path, the Euler step,
  the standalone per-equation helpers (kept as the trusted oracle), and the
  flat adaptive `_wcsph_dt_factors` traversal are unchanged; migrating them onto
  the generator is the ADR-0003 follow-up.
- ADR-0004 adds `neighbor_mode='grid'` to the generator: the same fused kernel
  body, but the flat `starts/lengths/neighbors` loop is replaced by a direct
  uniform-grid cell-list walk with the support cutoff inline (geometry split
  pre/post cutoff; the cell-block walk wraps the equation snippets via
  `_reindent`; `neighbor_mode` is part of the structural cache key). The
  continuity hot path (`compute_wcsph_accel_continuity` and
  `compute_wcsph_adaptive_timestep` via hand-written grid-direct
  `_wcsph_dt_factors_grid_{f32,f64}`) now defaults to grid mode and builds no
  flat CSR neighbor list; `neighbor_mode='flat'` is retained for the oracle and
  host-query paths. This is the gpu-nnps cache-build optimization landing on the
  backend; see the gpu-nnps aspect for the parity/perf evidence.
- ADR-0003 follow-up (generator migration): `warp_codegen` is now the single
  source for every neighbor-loop kernel; the ~14 duplicated hand `@wp.kernel`s
  (`_summation_density`, `_continuity`, `_pressure_gradient`,
  `_artificial_viscosity`, `_xsph_correction`, `_wcsph_dt_factors{,_grid}`,
  f32+f64) are retired. New: `accumulate_outputs` (seed `_acc_<out>` from the
  existing `d_<out>[i]` for read-modify-write composition, used by the standalone
  additive viscosity); `SummationDensity` and `WcsphCflFactor` blocks (the CFL
  `dt_cfl` is a free-form `wp.max` neighbor reduction, `dt_force` a per-particle
  `post_loop`); a shared `_run_equation_group` launcher (flat/grid, accumulate,
  cross-array). The standalone helpers keep their signatures/cross-array and now
  reject non-canonical `out_prop`/`out_props`. The generated kernel `func_name`
  is now a deterministic md5 of the structural cache key (was
  `len(_KERNEL_CACHE)`), so the source is byte-stable across runs and Warp's
  on-disk kernel cache hits instead of recompiling cold every session. EOS,
  integrator, and dt init/reduce/finalize reduction kernels are unchanged.
- 3D dam-break WCSPH physics (ADR-0005, additive). New device funcs: Wendland
  quintic value + dwdq (`dim==1/2/3` normalization) routed through the shared
  `_kernel_value`/`_kernel_dwdq` routers as kernel id 2 (cubic 0 / gaussian 1
  branches untouched); `_apply_body_force` (ramped gravity into au/av/aw under
  `dim>1`/`dim>2` guards); `_tait_eos_hg_correction` (clamp `rho>=rho0` so wall
  `p>=0`). New host driver `wc_sph_dam_break_step`: an E-P-E-C (== reference
  `EPECIntegrator`) continuity-density step that sums fluid accel + density over
  `[fluid, walls]`, takes wall density from `[fluid]` only, XSPH from fluid only,
  and holds walls fixed via zero accel through the shared `wcsph_pec_stage`. All
  composed from the existing generated equation blocks with
  `accumulate_outputs=True` over each source -- no new neighbor-loop code, no edit
  to ids 0/1, the generated 2D source, or the single-array elliptical-drop step.
  The 2D generated source is byte-identical (md5-pinned by
  `test_2d_path_generated_source_is_byte_identical_to_golden`); the router gained
  an additive `id==2` branch (logic-preserving; a one-time recompile can occur).
  Wendland inflates every generated kernel's PTX (~10-13 MB), so cold compiles are
  now multi-minute per large 3D fused kernel.
- Dam-break perf follow-up: the fluid acceleration+density blocks were **fused**
  via `_WCSPH_DAM_BREAK_FLUID_BLOCKS` (PressureGradient + ArtificialViscosity +
  ContinuityEquation) run as one generated kernel per source
  (`accumulate_outputs=True`) instead of three separate single-block launches --
  one neighbour walk / one per-pair geometry for all three. 1.23x faster Warp
  step at >1M (0.415 -> 0.337 s/step; ~13-15x vs single-thread CPU). XSPH
  (fluid-only) and wall continuity (fluid->wall) stay separate (heterogeneous
  source/destination), so it is not a single fused kernel like the single-array
  elliptical path. New cache entry; the 2D generated source stays byte-identical
  (guard `test_2d_path_generated_source_is_byte_identical_to_golden` passes).

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: warp-backend` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - device data ownership.
- Influences: `gpu-nnps` - backend-specific neighbor kernels.
