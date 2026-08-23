# 01 Mesh And Geometry

## Geometry Model

- [OBSERVED] PySPH particles are points assigned physical properties, and a homogeneous collection is represented by `ParticleArray` at `docs/source/design/working_with_particles.rst:7-13`.
- [OBSERVED] User-facing particle positions are ordinary properties such as `x`, `y`, and `z`; they are created through `get_particle_array()` and accessed as ParticleArray attributes at `docs/source/design/working_with_particles.rst:29-48`.
- [OBSERVED] Smoothing length `h` is a default double property documented with the other default properties at `docs/source/using_pysph.rst:78-89`.
- [INFERRED] ParticleArray itself is meshfree storage: it does not own cells, faces, connectivity, control volumes, or shape functions.

## Mesh Topology

- [OBSERVED] NNPS and domain managers build spatial structures after particles move; NNPS update computes bounds, refreshes its data structure, then bins particle indices at `pysph/base/nnps_base.pyx:1471-1506`.
- [INFERRED] Cell lists, octrees, and periodic/mirror ghost construction are downstream consumers of ParticleArray, not part of the ParticleArray data model.
- [UNKNOWN] It is not yet specified whether a Warp ParticleArray should expose geometry arrays directly to a future Warp NNPS or preserve the current ParticleArray/NNPS wrapper boundary first.

## Coordinate And Layout Assumptions

- [OBSERVED] Property data are flat arrays, and strided properties are represented as a flat array whose logical particle count is array length divided by stride at `docs/source/using_pysph.rst:94-104` and `pysph/base/particle_array.pyx:423-437`.
- [OBSERVED] Tests expect a stride-3 property with four particles to have flat length 12 and logical count 4 at `pysph/base/tests/test_particle_array.py:180-195`.
- [INFERRED] A Warp port should treat ParticleArray as structure-of-arrays with optional fixed per-particle stride, not as array-of-structs, unless a separate compatibility layer preserves flat property semantics.

## Geometry-Specific Non-Applicability

- [INFERRED] There are no face normals, finite-volume areas, element volumes, Riemann states, reconstruction stencils, or mesh boundary IDs in ParticleArray.
- [UNKNOWN] Geometry invariants required by all PySPH equations are not fully enumerated here; this spec only covers the base array layer.
