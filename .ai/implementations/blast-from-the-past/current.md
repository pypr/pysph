# Current - blast-from-the-past

Updated: 2026-07-06T10:49:00 CEST by codex

**Status:** The Warp backend has device-mirrored particle state, grid-direct
3D WCSPH, generated/fused equation groups, periodic neighbors, validated 2D
elliptical-drop and 3D dam-break cases, and ADR-0006 P0-P3 floating-rigid-body
support. P2 (`bb3843f7`) keeps moment reduction, 3x3 angular solve, RK2 state,
and rigid motion on the GPU. P3 adds deterministic two-pass Liu fluid/rigid
coupling, static rigid number density, device density/body-force staging, and
the sibling `wc_sph_dam_break_rigid_step` without modifying the fixed-wall
driver.

**Active aspects:** warp-backend, gpu-nnps, particle-memory,
validation-benchmarks, host-integration.

**In-flight experiment:**
`experiments/2026-06-19_warp-floating-body-rigid` (ADR-0006). P0-P3 are done;
P3 is approved by @prabhu and committed locally. P4 contact/long-horizon fidelity and the
photorealistic animation remain.

`experiments/2026-07-06_warp-adaptive-particle-resolution-p0` is active under
the approved dynamic-APR plan. The first checkpoint proves the existing
multi-solid driver can run a fixed Kleefsman obstacle: a warm 1,000-fluid +
3,824-wall + 4-obstacle case ran 250 steps to `t=0.258455`, remained finite,
developed 150.147 kPa maximum obstacle pressure, and kept obstacle device
coordinates bit-identical. Reference audit found the open PySPH adaptive code
is process-relevant but not a ready 3D/GPU implementation. A converged NumPy
kill test confirms icosahedron-plus-center beats cubic-plus-center, but does not
yet reproduce Vacondio's published Wendland error/mass ratio under the current
PySPH kernel convention; ADR-0007 remains intentionally deferred.

**Latest validation:** Final P3 Warp SPH suite: `54 passed, 2 warnings`.
The 7,458-particle coupled first-plunge transient ran 241 steps to `t=0.200603`,
remained finite with device error 0, moved/rotated the body from computed fluid
reaction, and preserved relative geometry to `1.90e-6`. The review image and
metrics are in
`reviews/2026-06-20_warp-liu-fluid-rigid-coupling-p3.md`.

**Open approvals:** P3 and 3D dam-break reviews are approved by @prabhu. PR
#435 remains an upstream publication item, not a local review blocker.

**Next action:** Detailed plan
`plans/2026-07-06_warp-multilevel-gpu-nnps.md` is pending owner approval for
the first runtime APR GPU milestone. P0 stencil-convention reconciliation and
uniform obstacle baselines remain prerequisites for production APR weights,
but do not block synthetic exact-set multilevel-NNPS work. Floating-body P4
remains queued.

**Known validation limitation:** Compyle 0.9.1 on Python 3.14 cannot run the
shipped CPU rigid Application (`ast.Str` removal). P3 uses direct NumPy
primitive parity plus end-to-end GPU smoke/transient validation; the review
discloses that a monolithic hand-staged EPEC oracle was not added.
