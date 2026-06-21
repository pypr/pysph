# Current - blast-from-the-past

Updated: 2026-06-21T02:06:46 CEST by codex

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

**Latest validation:** Final P3 Warp SPH suite: `54 passed, 2 warnings`.
The 7,458-particle coupled first-plunge transient ran 241 steps to `t=0.200603`,
remained finite with device error 0, moved/rotated the body from computed fluid
reaction, and preserved relative geometry to `1.90e-6`. The review image and
metrics are in
`reviews/2026-06-20_warp-liu-fluid-rigid-coupling-p3.md`.

**Open approvals:** P3 and 3D dam-break reviews are approved by @prabhu. PR
#435 remains an upstream publication item, not a local review blocker.

**Next action:** Plan P4: rigid-wall contact, longer EPEC/geometry-drift
validation, assembled
surge-tosses-a-box checkpoints, and `splashsurf -> Blender Cycles -> ffmpeg`
rendering.

**Known validation limitation:** Compyle 0.9.1 on Python 3.14 cannot run the
shipped CPU rigid Application (`ast.Str` removal). P3 uses direct NumPy
primitive parity plus end-to-end GPU smoke/transient validation; the review
discloses that a monolithic hand-staged EPEC oracle was not added.
