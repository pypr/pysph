# 03 Data Structures

## ParticleArray Object

- [OBSERVED] `ParticleArray` is a Cython extension class declared with `backend`, `properties`, `property_arrays`, `stride`, `output_property_arrays`, `constants`, `default_values`, `name`, `num_real_particles`, `lb_props`, `gpu`, and `time` fields at `pysph/base/particle_array.pxd:37-73`.
- [OBSERVED] `properties` maps property names to `BaseArray` subclasses, while `constants` maps names to fixed-size arrays at `pysph/base/particle_array.pyx:109-157` and `pysph/base/particle_array.pyx:823-850`.
- [OBSERVED] `default_values` stores per-property fill values used when adding/extending particles at `pysph/base/particle_array.pyx:531-602` and `pysph/base/particle_array.pyx:664-689`.
- [OBSERVED] `stride` stores per-property logical width and defaults to 1 when absent, as used by particle count, get, add, append, align, copy, and resize paths at `pysph/base/particle_array.pyx:423-437`, `pysph/base/particle_array.pyx:704-765`, `pysph/base/particle_array.pyx:531-602`, `pysph/base/particle_array.pyx:604-662`, `pysph/base/particle_array.pyx:1092-1173`, and `pysph/base/particle_array.pyx:1438-1449`.

## Property Storage

- [OBSERVED] `add_property()` creates properties from explicit data, scalar defaults, or empty arrays; it validates sizes against existing particle count and stride at `pysph/base/particle_array.pyx:851-1016`.
- [OBSERVED] `_create_carray()` maps type strings to `DoubleArray`, `LongArray`, `FloatArray`, `IntArray`, and `UIntArray`, then fills defaults at `pysph/base/particle_array.pyx:1020-1055`.
- [OBSERVED] `_create_c_array_from_npy_array()` maps NumPy int32/int64 to `LongArray`, float32 to `FloatArray`, and double to `DoubleArray` at `pysph/base/particle_array.pyx:1065-1090`.
- [OBSERVED] Tests verify scalar broadcasting, flattened 2D input, strided property length, and typed int properties at `pysph/base/tests/test_particle_array.py:150-178`, `pysph/base/tests/test_particle_array.py:469-533`, and `pysph/base/tests/test_particle_array.py:1020-1030`.
- [INFERRED] Property storage is SoA-like because each named property has its own contiguous array; strided properties are contiguous per property, not interleaved across property names.

## Baseline Properties And Tags

- [OBSERVED] `clear()` resets ParticleArray to contain `tag`, `pid`, and `gid` properties with defaults at `pysph/base/particle_array.pyx:395-400`.
- [OBSERVED] The `ParticleTag` enum defines `Local=0`, `Remote=1`, and `Ghost=2` at `pysph/base/particle_array.pxd:24-28`.
- [OBSERVED] Utility wrappers expose local/remote/ghost tag values through `ParticleTAGS` at `pysph/base/utils.py:15-20`.
- [OBSERVED] User docs describe `gid` as a globally unique index for load balancing, `pid` as the processor id, and `tag` as an integer used for local/remote/ghost classification at `docs/source/using_pysph.rst:106-132` and `docs/source/using_pysph.rst:300-308`.

## Constants

- [OBSERVED] Constants are added with `add_constant()`, cannot clash with existing property/constant names, are raveled into a carray, and are mirrored to GPU helpers if present at `pysph/base/particle_array.pyx:823-850`.
- [OBSERVED] Tests verify constants can be added in the constructor, read through `get()`, updated with `set()`, retrieved through `get_carray()`, cloned, and kept fixed when particles are added at `pysph/base/tests/test_particle_array.py:758-845` and `pysph/base/tests/test_particle_array.py:847-884`.
- [INFERRED] A Warp port should preserve constants as named, non-particle-count-sized arrays because equations and output code may treat them separately from properties.

## DeviceHelper Mirror

- [OBSERVED] `DeviceHelper` stores a reference to the ParticleArray, a backend name, dtype policy, `num_real_particles`, and `_data` mapping, then creates device arrays for each property and constant at `pysph/base/device_helper.py:56-70`.
- [OBSERVED] `_get_array()` converts float/double arrays to the configured float precision and preserves integer dtype before creating a compyle `Array` at `pysph/base/device_helper.py:72-93`.
- [OBSERVED] `update_prop()` and `update_const()` keep `properties`, `constants`, `_data`, and ParticleArray host metadata in sync at `pysph/base/device_helper.py:144-179`.
- [OBSERVED] DeviceHelper tests cover mirror creation, selective/full push, selective/full pull, min/max, property add/remove, resize/extend/remove, align, append, empty clone, and extract at `pysph/base/tests/test_device_helper.py:29-403`.
- [INFERRED] DeviceHelper is the closest existing abstraction boundary for a Warp-backed mirror, but its compyle-specific `Array` type and kernel generation are implementation details to isolate.

## Serialization Schema

- [OBSERVED] `get_particles_info()` records each property name, c type, default, stride, and null data placeholder, plus constants, output arrays, and load-balance properties at `pysph/base/utils.py:466-497`.
- [OBSERVED] `create_dummy_particles()` reconstructs empty ParticleArray replicas from that metadata at `pysph/base/utils.py:500-512`.
- [OBSERVED] Output loaders reconstruct ParticleArray from saved property metadata and constants at `pysph/solver/output.py:127-162` and `pysph/solver/output.py:195-221`.
- [INFERRED] Any Warp migration must preserve this schema unless output and MPI dummy creation are migrated at the same time.
