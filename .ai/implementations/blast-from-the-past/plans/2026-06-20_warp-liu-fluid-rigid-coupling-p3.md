---
type: plan
id: 2026-06-20_warp-liu-fluid-rigid-coupling-p3
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-20T19:08:57 CEST
status: approved
aspects: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks]
adr: ADR-0006
within_boundary: true
host_files:
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_sph.py
---

# Plan: Liu fluid–rigid coupling and sibling dam-break driver (ADR-0006 P3)

## Goal

Couple the existing device-resident rigid-body integrator to WCSPH fluid using
the real PySPH `LiuFluidForce` formulation, then run the first physically
meaningful Warp GPU case in which a dam-break surge moves a floating box.
Keep particle and compact rigid state device-resident, preserve the existing
fixed-wall driver unchanged, and produce numerical plus visual evidence.

## Context

P2 can integrate force/torque entirely on the GPU, but it currently receives
prescribed forces. P3 must generate those forces from SPH interactions.

Source audit found two details that sharpen the old handoff:

- `LiuFluidForce` itself does **not** consume `V`, although its historical
  signature includes it. `NumberDensity` remains useful as a one-time static
  body-volume pre-pass and parity diagnostic, but it must not be presented as
  numerically required by Liu or rebuilt every step.
- A single fluid-destination Liu kernel would need atomic writes into source
  body force arrays. fp32 atomic sums were already rejected by P0 as
  nondeterministic. P3 will therefore use two deterministic generated passes:
  fluid acceleration (`body -> fluid`) and the equal-and-opposite body reaction
  (`fluid -> body`, destination-register accumulation). This duplicates one
  neighbor walk but avoids source atomics and gives each body particle a stable
  force before the existing f64 rigid reduction.

## Approach

### 1. Additive Liu equation blocks

- Add `LiuFluidAcceleration(WarpEquation)`:
  destination fluid, source body, outputs only fluid `au/av/aw`, implementing
  `-m_body * (p_body/rho_body^2 + p_fluid/rho_fluid^2) * DWIJ`.
- Add `LiuBodyReaction(WarpEquation)`:
  destination body, source fluid, outputs body `fx/fy/fz`; use the reversed
  gradient sign so each interaction is equal-and-opposite to fluid momentum.
- Add `NumberDensity(WarpEquation)` (`V = sum W`) for a one-time self-neighbor
  body pre-pass. Keep it outside the stepping hot path because rigid relative
  geometry—and therefore `V`—does not change.
- All blocks are additive new structural cache entries. Do not edit existing
  generated equation source, kernel routers, or `_WCSPH_DAM_BREAK_FLUID_BLOCKS`.

### 2. Rigid density and force staging

- Add a device density save/stage path for the rigid body:
  `rho0 <- rho` once per physical step, then
  `rho = rho0 + stage*dt*arho` at midpoint/full stages. This updates body
  pressure response without sending the body through `wcsph_pec_stage`, which
  would incorrectly integrate its particles as independent fluid points.
- Add a device body-force initializer that zeros `fx/fy/fz` and seeds
  `m*[gx,gy,gz]` before Liu reaction accumulation.
- Body EOS is Tait-HG; body continuity comes from fluid neighbors only, matching
  the shipped rigid examples' pressure-response pattern.

### 3. New sibling driver: `wc_sph_dam_break_rigid_step`

- Inputs: fluid index, fixed-wall indices, rigid-body index/state, dt/physics
  controls. Never edit `wc_sph_dam_break_step`.
- Save fluid/wall WCSPH state, body density state, and P2 rigid state once.
- At each EPEC force evaluation:
  1. EOS: fluid Tait; fixed walls and body Tait-HG.
  2. Zero fluid/wall outputs; initialize body force with gravity.
  3. Existing fused pressure+AV+continuity over fluid and fixed walls only.
  4. Body contribution to **fluid continuity exactly once** via a standalone
     `ContinuityEquation` pass—this resolves the recorded `arho` double-count.
  5. Liu fluid acceleration and deterministic body reaction passes.
  6. Body continuity from fluid; wall continuity from fluid; fluid-only XSPH;
     fluid gravity.
- Predictor: fluid/walls PEC half-stage; body density half-stage; P2 rigid RK2
  half-stage; update NNPS from device positions.
- Corrector: reevaluate all forces at midpoint, then full fluid/wall, body
  density, and rigid-body stages; update NNPS again.
- Adaptive dt remains the fluid device reduction plus the existing scalar
  handoff. A rigid/contact dt criterion is deferred unless the transient proves
  it necessary.

### 4. First coupled runner and visual checkpoint

- Extend `experiments/2026-06-19-warp-floating-body-rigid/` with a runner for a
  rectangular floating box in the validated 3D dam-break flume.
- Start coarse enough for rapid transient iteration, then run a medium case to
  the first surge/body impact. Save checkpoints for body COM/orientation proxy,
  linear/angular velocity, force/torque, fluid density bounds, energy, and
  finiteness.
- Produce a review image with the actual fluid particles/surface plus the box;
  label it as a scientific render. Photorealistic Blender animation remains P4
  because Blender/ffmpeg are absent on this host.

## Files expected to change

- `pysph/base/warp_sph.py` — additive Liu/NumberDensity blocks, rigid density
  stage, body-force initialization, and sibling coupled driver.
- `pysph/base/tests/test_warp_sph.py` — pairwise parity, equal/opposite force,
  staging/order, existing-driver guard, and first-transient tests.
- `.ai/implementations/blast-from-the-past/experiments/2026-06-19-warp-floating-body-rigid/`
  — coupled runner, metrics, and visual artifact.
- ADR/plan/review/session/aspect/current memory required by the contract.

## Tests / validation

- `NumberDensity` self-neighbor values match a CPU Wendland reference.
- One fluid/body pair and asymmetric many-particle fixture match a direct NumPy
  Liu reference in fp32/fp64.
- Total fluid force plus body reaction is zero at interaction precision; body
  torque matches direct `sum(r x f)`.
- Determinism: repeated body-reaction runs are stable because there are no
  source atomics; P2 f64 reduction remains stable.
- Rigid density midpoint/full stages match NumPy and do not alter rigid
  position/velocity independently.
- One coupled EPEC step matches a hand-rolled CPU/NumPy stage oracle for fluid
  acceleration/density, body density, reaction force/torque, COM velocity,
  angular velocity, and positions.
- First-plunge transient: initially supported/near-rest body then incoming
  surge; check finite state, physically directed vertical/horizontal response,
  body geometry, momentum reaction, and reproducibility. Do not use static
  Archimedes draft as the sole gate—it cannot distinguish coupling models.
- Existing `wc_sph_dam_break_step` behavior test passes unchanged; 2D generated
  source golden guard remains byte-identical.
- Focused suite, final warm full Warp SPH suite, `validate-memory.py`,
  `git diff --check`, and adversarial review before @prabhu sign-off.

## Success criteria

- A dam-break surge moves and rotates the box from computed Liu forces, with no
  per-stage particle-state host transfer and no body `arho` double-count.
- Fluid/body interaction force is equal-and-opposite within dtype tolerance;
  body force/torque and one-step state match the direct oracle.
- Existing fixed-wall and 2D paths remain unchanged.
- A medium coupled run remains finite through first impact and yields an actual
  review image plus quantitative metrics.

## Risks

- Two-pass coupling is deterministic but traverses body/fluid neighbors twice;
  profile after correctness before considering a mixed-dtype atomic/fused path.
- Independently accumulated equal/opposite passes can differ at fp32 rounding
  scale; gate total momentum with a dtype-derived tolerance.
- Body density/EOS staging order is easy to get subtly wrong; one-step and
  midpoint oracle tests pin it.
- No collision/contact force is included initially. Choose the first-impact
  horizon so the body does not penetrate a fixed wall; add the already-deferred
  wall collision only if the case requires it.
- P2 is implemented but cannot be committed until @prabhu gives exact `LGTM`
  on its review. P3 should remain an uncommitted stacked change until that gate
  clears, or start only after sign-off if review isolation is preferred.

## Out of scope

- Photorealistic Blender/Cycles animation and ffmpeg stitching (P4).
- Rigid-wall or rigid-rigid contact unless required by the selected transient.
- Quaternion/orientation-matrix integration; P4 evaluates measured drift first.
- Compyle/Python 3.14 repair and full shipped CPU rigid Application parity.
- Coupling-kernel fusion or CUDA graph capture.

## Estimated effort

One to two substantial implementation sessions, approximately 350-550 LOC in
backend/tests plus the experiment runner and review artifacts.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-20T19:13:48 CEST
- Approval, verbatim quote:
  > take this as @prabhu: LGTM for P2. approved for P3
