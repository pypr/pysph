# 06 Host Contract

## Public API Surface

- [OBSERVED] The declared Cython API includes methods for time, naming, properties, constants, load-balance properties, particle counts, mutation, alignment, cloning, extraction, copy, zeroing, pid, min/max, and resize at `pysph/base/particle_array.pxd:75-138`.
- [OBSERVED] Attribute access maps known property/constant names to arrays and raises `AttributeError` for missing names at `pysph/base/particle_array.pyx:159-177`; tests cover attribute reads and writes at `pysph/base/tests/test_particle_array.py:242-288`.
- [OBSERVED] `get()` returns selected properties/constants as NumPy arrays or tuples, defaults to only real particles, and returns arrays that do not own their data at `pysph/base/particle_array.pyx:704-765`.
- [OBSERVED] `get_carray()` returns the underlying carray for a property or constant and raises `KeyError` otherwise at `pysph/base/particle_array.pyx:811-821`.
- [INFERRED] Existing Cython callers may require `BaseArray` objects from `get_carray()`, so a Warp port that replaces host storage must either preserve these objects or introduce a compatibility boundary with a separate plan.

## Mutation Semantics

- [OBSERVED] `remove_particles()` removes selected indices, handles stride, validates oversized removal, and can align afterward; tests cover normal, strided, oversized, and out-of-range cases at `pysph/base/particle_array.pyx:439-505` and `pysph/base/tests/test_particle_array.py:290-337`.
- [OBSERVED] `add_particles()` appends supplied property values, fills omitted properties with defaults, supports empty adds, and can align afterward at `pysph/base/particle_array.pyx:531-602` and `pysph/base/tests/test_particle_array.py:339-399`.
- [OBSERVED] `append_parray()` appends another ParticleArray, adds missing properties with defaults, optionally updates constants, and can align afterward at `pysph/base/particle_array.pyx:604-662` and `pysph/base/tests/test_particle_array.py:678-718`.
- [OBSERVED] `extract_particles()` creates or populates a destination ParticleArray, extends it, copies selected strided values, and can align afterward at `pysph/base/particle_array.pyx:1237-1320` and `pysph/base/tests/test_particle_array.py:886-979`.
- [OBSERVED] `resize()` resizes all property arrays but does not update the particle count until alignment/length semantics are applied by callers at `pysph/base/particle_array.pyx:1438-1449`.

## Synchronization Contract

- [OBSERVED] `get_property_arrays()` pulls requested device properties before returning host arrays when `backend` is not `cython` at `pysph/base/particle_array.pyx:344-386`.
- [OBSERVED] ParticleArray exposes `set_device_helper()` to replace or attach a helper at `pysph/base/particle_array.pyx:767-770`.
- [OBSERVED] DeviceHelper tests require host changes to become visible on device after `push()` and device changes to become visible on host after `pull()` at `pysph/base/tests/test_device_helper.py:51-141`.
- [INFERRED] Warp integration must make host/device authority explicit for every API that returns host arrays, modifies device arrays, or serializes particles.

## Compatibility Constraints

- [OBSERVED] Pickle roundtrip stores properties/defaults/stride/constants and restores `num_real_particles` by counting Local tags at `pysph/base/particle_array.pyx:179-224`; tests cover pickle roundtrip at `pysph/base/tests/test_particle_array.py:1048-1066`.
- [OBSERVED] `remove_property()` also removes the property from output arrays and delegates to the GPU helper if present at `pysph/base/particle_array.pyx:1412-1421`; tests cover this output-array side effect at `pysph/base/tests/test_particle_array.py:980-991`.
- [OBSERVED] `empty_clone()` preserves constants, selected properties, name, and output arrays at `pysph/base/particle_array.pyx:1174-1215`; tests cover clone behavior at `pysph/base/tests/test_particle_array.py:847-884`.
- [INFERRED] The first Warp migration should keep the Python/Cython public API behavior stable before attempting broader solver-facing changes.
