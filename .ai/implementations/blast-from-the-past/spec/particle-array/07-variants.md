# 07 Variants

## Backend Variants

- [OBSERVED] ParticleArray resolves the configured backend with `get_backend(backend)` and treats `cython` as the host-only backend at `pysph/base/particle_array.pyx:109-116` and `pysph/base/particle_array.pyx:149-157`.
- [OBSERVED] Tests define CPU, OpenCL, and CUDA ParticleArray variants through `ParticleArrayTestCPU`, `ParticleArrayTestOpenCL`, and `ParticleArrayTestCUDA` at `pysph/base/tests/test_particle_array.py:1033-1139`.
- [OBSERVED] DeviceHelper tests parametrize `cython`, `opencl`, and `cuda` where imports are available at `pysph/base/tests/test_device_helper.py:18-26`.
- [UNKNOWN] A Warp variant name, configuration path, and fallback behavior are not yet defined.

## Storage Variants

- [OBSERVED] Properties may be scalar-stride or fixed-stride, with strided properties stored as flat arrays; tests cover stride 2 and stride 3 properties at `pysph/base/tests/test_particle_array.py:123-149`, `pysph/base/tests/test_particle_array.py:180-195`, and `pysph/base/tests/test_particle_array.py:573-587`.
- [OBSERVED] Properties may be double, float, int, long, or unsigned int according to carray creation paths at `pysph/base/particle_array.pyx:1020-1055`.
- [OBSERVED] Constants are stored outside `properties` and may be scalar or vector arrays at `pysph/base/tests/test_particle_array.py:758-845`.
- [INFERRED] Warp storage must model at least three cases: scalar per-particle arrays, fixed-width strided per-particle arrays, and fixed-size constants.

## Output/Readback Variants

- [OBSERVED] `get_property_arrays(all=False, only_real=True)` can return output arrays only or all arrays, and can slice only real particles or all particles at `pysph/base/particle_array.pyx:344-386`.
- [OBSERVED] Output code passes `detailed_output` and `only_real` into `get_property_arrays()` at `pysph/solver/output.py:53-78`.
- [INFERRED] A Warp migration should preserve explicit all-vs-output and real-vs-all readback modes because they affect I/O size and correctness.

## Ordering Variants

- [OBSERVED] CPU tests generally expect deterministic post-mutation ordering for aligned arrays at `pysph/base/tests/test_particle_array.py:589-676`.
- [OBSERVED] Existing GPU behavior is already allowed to differ from CPU ordering in one tagged-removal/strided-property case at `pysph/base/tests/test_particle_array.py:452-460`.
- [UNKNOWN] The future Warp backend's exact ordering guarantees need a decision before tests are written.
