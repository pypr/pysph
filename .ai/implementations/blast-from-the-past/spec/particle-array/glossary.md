# Glossary - Particle Array

- [OBSERVED] `ParticleArray`: Homogeneous collection of particles and named properties; documented as the main data structure in `docs/source/using_pysph.rst:59-76`.
- [OBSERVED] `BaseArray`: Cython array abstraction used for ParticleArray properties; docs name `cyarray.carray.BaseArray` as the storage family at `docs/source/using_pysph.rst:59-76`.
- [OBSERVED] Property: A named one-dimensional array associated with particles; arbitrary properties are supported at `docs/source/using_pysph.rst:59-76`.
- [OBSERVED] Stride: Fixed number of flat array entries per logical particle for a property; documented at `docs/source/using_pysph.rst:94-104`.
- [OBSERVED] Constant: Fixed-size array associated with a ParticleArray, not resized with particles; documented at `docs/source/using_pysph.rst:148-175`.
- [OBSERVED] `Local`: Particle tag value 0 in the `ParticleTag` enum at `pysph/base/particle_array.pxd:24-28`.
- [OBSERVED] `Remote`: Particle tag value 1 in the `ParticleTag` enum at `pysph/base/particle_array.pxd:24-28`.
- [OBSERVED] `Ghost`: Particle tag value 2 in the `ParticleTag` enum at `pysph/base/particle_array.pxd:24-28`.
- [OBSERVED] `num_real_particles`: Count of Local-tagged particles after alignment; updated in `align_particles()` at `pysph/base/particle_array.pyx:1092-1173`.
- [OBSERVED] `DeviceHelper`: Existing device mirror for ParticleArray properties/constants, described by its class docstring and constructor at `pysph/base/device_helper.py:47-70`.
- [OBSERVED] `lb_props`: Load-balance property list recorded by `get_particles_info()` and restored in dummy particles at `pysph/base/utils.py:466-512`.
- [INFERRED] Warp mirror: Proposed future object that would provide DeviceHelper-like semantics using NVIDIA Warp arrays/kernels.
