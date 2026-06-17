---
aspect: warp-backend
implementation: blast-from-the-past
owner: @kunalpuri-prediqt
created: 2026-06-15T07:19:08 CET
last_reviewed: 2026-06-15T08:34:00 CET
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

## Key sub-topics

- Warp version/API surface - Active environment has Warp `1.14.0`; confirm documentation set with team.
- Kernel launch model for partition/gather/scatter kernels - Confirm with prototype.
- Compatibility with existing PySPH GPU pathways.
- Backend naming and ownership ADR.
- Next kernel family decision: move add/remove/extract/append growth internals from host-side NumPy concatenation to fully device-side Warp kernels.
- Tutorial documentation added at `docs/source/tutorial/warp_particle_array.rst`.
- Adaptive timestep reductions currently transfer only the final scalar `dt`
  back to Python because launch parameters remain host scalars.
- The PySPH Application parity step uses device-side saved state plus
  continuity-density PEC stages; the original summation-density KDK path remains
  the compatibility default.

## References for this aspect

- `.ai/implementations/blast-from-the-past/references/index.md`

## Decisions affecting this aspect

- Filter `.ai/implementations/blast-from-the-past/decisions/index.json` for `scope: warp-backend` and `scope: global`.

## Cross-aspect dependencies

- Depends on: `particle-memory` - device data ownership.
- Influences: `gpu-nnps` - backend-specific neighbor kernels.
