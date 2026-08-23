# Open Questions - Particle Array Spec

This file is intentionally first: the attached spec prompt requires unknowns to be surfaced before turning the current implementation into a porting contract.

## Questions

- [UNKNOWN] OQ-PA-001: Should a Warp-backed particle array preserve PySPH's current `backend` names (`cython`, `opencl`, `cuda`) or add a new `warp` backend string? The current constructor resolves `backend` through `get_backend(backend)` and stores it on `self.backend` at `pysph/base/particle_array.pyx:109-116`.
- [UNKNOWN] OQ-PA-002: Should Warp own the authoritative storage for migrated arrays, or should Warp remain a mirror of host `cyarray` data like `DeviceHelper`? The current design creates host `BaseArray` objects in `ParticleArray` and device mirrors in `DeviceHelper` when `backend != 'cython'` at `pysph/base/particle_array.pyx:109-157` and `pysph/base/device_helper.py:47-70`.
- [UNKNOWN] OQ-PA-003: What is the required dtype policy for Warp: always double when PySPH `use_double` is true, or per-property dtype parity with `BaseArray`? Existing `DeviceHelper` converts float/double arrays according to `get_config().use_double` while preserving integer dtype at `pysph/base/device_helper.py:47-77`.
- [UNKNOWN] OQ-PA-004: Which ParticleArray operations must be fast on GPU in the first implementation: allocation, push/pull, mutation, alignment, append/extract/remove, or NNPS-facing access? The current tests cover all of these across CPU and GPU helpers at `pysph/base/tests/test_particle_array.py:290-1018` and `pysph/base/tests/test_device_helper.py:51-403`.
- [UNKNOWN] OQ-PA-005: Should the first Warp port preserve exact post-operation ordering for ghost/tagged/strided arrays, or only preserve semantic equivalence? Existing GPU ordering already differs from CPU for one strided tagged-removal assertion at `pysph/base/tests/test_particle_array.py:452-460`.
- [UNKNOWN] OQ-PA-006: What is the minimal accepted integration boundary for `.pxd/.pyx` callers? `ParticleArray` is a Cython extension type with declared cpdef/cdef methods at `pysph/base/particle_array.pxd:37-138`, so replacing it wholesale may affect ABI expectations.
- [UNKNOWN] OQ-PA-007: Should constants become Warp arrays, scalar Python/NumPy state, or both? Current constants are stored as `BaseArray` objects, are not resized with particles, and are mirrored to GPU helpers at `pysph/base/particle_array.pyx:823-850` and `pysph/base/tests/test_particle_array.py:758-845`.
- [UNKNOWN] OQ-PA-008: Should Warp expose direct arrays to NNPS and equations, or should it support existing host pull/readback points first? NNPS consumes ParticleArray data through wrappers and `get_number_of_particles()` at `pysph/base/nnps_base.pyx:1459-1510`.
- [UNKNOWN] OQ-PA-009: What measurable performance target defines "blazing fast particle dynamics" for this first ParticleArray migration? The implementation memory currently records the success criterion qualitatively, not as a benchmark threshold.
