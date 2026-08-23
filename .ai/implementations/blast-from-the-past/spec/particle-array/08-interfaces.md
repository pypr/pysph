# 08 Interfaces

## Existing Interfaces To Preserve

- [OBSERVED] Python construction accepts either raw arrays or dictionaries containing `data`, `type`, `default`, and `stride`-like metadata; tests cover both forms at `pysph/base/tests/test_particle_array.py:50-83`, `pysph/base/tests/test_particle_array.py:112-149`, and `pysph/base/tests/test_particle_array.py:150-178`.
- [OBSERVED] `get_particle_array()` is the common helper that adds default properties and output metadata at `pysph/base/utils.py:47-149`.
- [OBSERVED] Attribute access, `get()`, `set()`, `get_carray()`, `add_property()`, `remove_property()`, `add_constant()`, and mutation methods are public behaviors exercised by tests at `pysph/base/tests/test_particle_array.py:242-288`, `pysph/base/tests/test_particle_array.py:197-217`, `pysph/base/tests/test_particle_array.py:1068-1103`, `pysph/base/tests/test_particle_array.py:834-845`, `pysph/base/tests/test_particle_array.py:469-533`, `pysph/base/tests/test_particle_array.py:980-991`, and `pysph/base/tests/test_particle_array.py:758-832`.

## Solver-Agnostic Interface Decomposition

- [INFERRED] `ParticleSchema`: names, dtypes, defaults, strides, output-array membership, load-balance membership, and constant metadata.
- [INFERRED] `HostParticleStore`: host `BaseArray` ownership, NumPy/carray views, serialization, pickle, and Cython ABI compatibility.
- [INFERRED] `DeviceParticleMirror`: backend-specific arrays plus push/pull/update hooks; existing `DeviceHelper` is the observed compyle implementation at `pysph/base/device_helper.py:47-70`.
- [INFERRED] `ParticleSelectionOps`: remove, extract, append, align, copy, and strided gather/scatter kernels.
- [INFERRED] `ParticleLifecycleOps`: construction, extend, resize, defaults fill, constants preservation, and metadata reconstruction.

## Proposed Warp Boundary

- [INFERRED] First Warp integration should likely implement a `DeviceParticleMirror` equivalent rather than replacing `ParticleArray` host storage, because host carray access is part of the declared and tested API.
- [INFERRED] Warp kernels should be introduced behind methods equivalent to `DeviceHelper.align_particles()`, `remove_particles()`, `remove_tagged_particles()`, `add_particles()`, `append_parray()`, `extend()`, `extract_particles()`, `resize()`, `push()`, and `pull()`.
- [UNKNOWN] Whether this boundary lives in a new helper class, an extension of `DeviceHelper`, or a separate `warp_device_helper.py` requires an ADR before code migration.

## Required Error Behavior

- [OBSERVED] Missing properties accessed as attributes raise `AttributeError` at `pysph/base/particle_array.pyx:159-177`, and tests assert this at `pysph/base/tests/test_particle_array.py:242-265`.
- [OBSERVED] `remove_particles()` raises `ValueError` when asked to remove more indices than particles at `pysph/base/particle_array.pyx:439-505`, with tests at `pysph/base/tests/test_particle_array.py:290-337`.
- [OBSERVED] `set()` raises for unknown names and delegates incompatible length handling to carray set-data behavior; tests cover longer data raising `ValueError` at `pysph/base/particle_array.pyx:772-810` and `pysph/base/tests/test_particle_array.py:1068-1103`.
- [INFERRED] Warp code must preserve Python-visible exceptions at the ParticleArray API boundary even if device kernels use different internal failure modes.
