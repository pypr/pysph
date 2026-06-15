# 10 Porting

## Porting Premises

- [OBSERVED] The implementation is scoped to `blast-from-the-past`, and its boundary includes `pysph/**/*.pxd` and `pysph/**/*.pyx` at `.ai/implementations/blast-from-the-past/implementation.md:2-6` and `.ai/implementations/blast-from-the-past/implementation.md:27-33`.
- [OBSERVED] `ParticleArray` is declared in `.pxd` and implemented in `.pyx`, with GPU mirroring delegated through Python `DeviceHelper` at `pysph/base/particle_array.pxd:37-138`, `pysph/base/particle_array.pyx:109-157`, and `pysph/base/device_helper.py:47-70`.
- [INFERRED] The lowest-risk Warp migration is additive: introduce Warp as a mirror/backend path while preserving `ParticleArray` host storage and Cython declarations.

## Phase 2 Plan Seed

1. [UNKNOWN] Decide backend naming and ownership in an ADR: `warp` backend string versus CUDA backend replacement, and mirror versus authoritative storage.
2. [INFERRED] Add a Warp helper behind the existing DeviceHelper-like interface with no changes to solver code.
3. [INFERRED] Implement push/pull and device-array creation first, because they validate dtype/stride/constant layout without mutation complexity.
4. [INFERRED] Add device alignment next, because Local-first partition controls `num_real_particles`, real-only slices, and NNPS expectations.
5. [INFERRED] Add remove/extract/append/extend kernels after alignment, because they require gather/scatter and resize/fill semantics.
6. [INFERRED] Only after ParticleArray behavior passes should NNPS or solver-loop Warp integration begin.

## Data Mapping Checklist

- [OBSERVED] Property dtype mapping must cover double, float, int, long, and unsigned int as created by `_create_carray()` at `pysph/base/particle_array.pyx:1020-1055`.
- [OBSERVED] Stride mapping must preserve logical particle count as `length / stride` at `pysph/base/particle_array.pyx:423-437`.
- [OBSERVED] Constants must not be resized during particle mutation at `pysph/base/tests/test_particle_array.py:805-817`.
- [OBSERVED] `num_real_particles` must be updated after tag alignment at `pysph/base/particle_array.pyx:1092-1173`.
- [OBSERVED] Output and dummy-particle metadata must preserve name, property type, default, stride, constants, output arrays, and load-balance properties at `pysph/base/utils.py:466-512`.

## Risks To Retire Before Code

- [UNKNOWN] Warp availability and version on `prediqt-02` are not yet confirmed.
- [UNKNOWN] Warp support for the required dtype matrix and efficient dynamic resizing strategy is not yet confirmed.
- [UNKNOWN] The performance target for "blazing fast particle dynamics" is not yet quantified.
- [INFERRED] Replacing host `BaseArray` storage too early risks breaking Cython callers and output/load paths.
- [INFERRED] Treating strided properties as separate vector types may break flat-array user expectations unless a compatibility view is maintained.
