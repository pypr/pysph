# 04 Boundary

## Boundary Concepts Present In ParticleArray

- [OBSERVED] ParticleArray has no geometric boundary-condition object; it stores boundary-relevant classification through `tag`, where `Local`, `Remote`, and `Ghost` are defined at `pysph/base/particle_array.pxd:24-28`.
- [OBSERVED] Documentation identifies Remote-tagged particles as particles assigned to but not owned by the processor, and Ghost-tagged particles as locally created boundary-condition particles at `docs/source/using_pysph.rst:300-308`.
- [OBSERVED] ParticleArray can remove particles by tag via `remove_tagged_particles()` at `pysph/base/particle_array.pyx:506-529`.
- [OBSERVED] Tests verify removing tagged particles for Local, Remote, and Ghost-like tag values, including strided properties, at `pysph/base/tests/test_particle_array.py:400-467`.

## Boundary Consumers

- [OBSERVED] Domain manager code removes old Ghost-tagged particles before creating periodic/mirror ghosts at `pysph/base/nnps_base.pyx:386-403` and `pysph/base/nnps_base.pyx:450-470`.
- [OBSERVED] Mirror-boundary handling extracts particles, modifies copied positions/velocities, and appends the generated particle arrays back into the original array at `pysph/base/nnps_base.pyx:520-610`.
- [INFERRED] Boundary condition generation depends on ParticleArray mutation primitives (`extract_particles`, `append_parray`, `remove_tagged_particles`) rather than on ParticleArray owning boundary-condition logic.

## Porting Boundary Rules

- [OBSERVED] `align_particles()` uses tag values to move Local particles to the beginning and updates `num_real_particles` at `pysph/base/particle_array.pyx:1092-1173`.
- [OBSERVED] `get()` defaults to returning only real/local particles, using `num_real_particles` as the slice bound at `pysph/base/particle_array.pyx:704-765`.
- [INFERRED] A Warp-backed boundary path must preserve the tag partition invariant before host code requests local-only slices, output with `only_real`, or NNPS-local indexing.
- [UNKNOWN] It is not yet decided whether boundary ghost generation itself should remain Cython/host-side while only storage primitives move to Warp.
