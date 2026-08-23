# 05 Parallelism

## MPI/Distributed State

- [OBSERVED] User docs describe `gid` as globally unique for parallel load balancing and `pid` as the processor id at `docs/source/using_pysph.rst:106-112`.
- [OBSERVED] `get_lb_props()` returns load-balance properties, defaulting to all properties when `lb_props` is empty at `pysph/base/particle_array.pyx:411-421`.
- [OBSERVED] Application particle creation broadcasts metadata from rank 0 and creates dummy particle arrays on non-root ranks at `pysph/solver/application.py:859-920`.
- [OBSERVED] `get_particles_info()` records `lb_props` and `create_dummy_particles()` restores them at `pysph/base/utils.py:466-512`.
- [INFERRED] A storage migration cannot ignore `gid`, `pid`, `tag`, or `lb_props` because they are part of distributed particle ownership and reconstruction.

## Device Parallelism

- [OBSERVED] Existing GPU backends are represented by non-cython `backend` values and use `DeviceHelper` to delegate operations such as alignment, removal, add, append, extend, extract, and resize at `pysph/base/particle_array.pyx:439-505`, `pysph/base/particle_array.pyx:531-602`, `pysph/base/particle_array.pyx:604-662`, `pysph/base/particle_array.pyx:664-689`, `pysph/base/particle_array.pyx:1092-1131`, `pysph/base/particle_array.pyx:1237-1277`, and `pysph/base/particle_array.pyx:1438-1445`.
- [OBSERVED] DeviceHelper implements alignment by generating index arrays and applying them to every property, with separate handling for strided properties at `pysph/base/device_helper.py:107-142` and `pysph/base/device_helper.py:249-323`.
- [OBSERVED] DeviceHelper implements particle removal by generating boolean masks and applying strided index maps at `pysph/base/device_helper.py:339-461`.
- [OBSERVED] DeviceHelper implements add/append/extend/extract operations on device arrays at `pysph/base/device_helper.py:463-672`.
- [INFERRED] Warp kernels will need equivalents for prefix/partition, gather/scatter, resize/fill, and strided copy primitives.

## Precision And Backend Variants

- [OBSERVED] GPU tests set `cfg.use_double = True` before testing OpenCL and CUDA ParticleArray backends at `pysph/base/tests/test_particle_array.py:1106-1139`.
- [OBSERVED] DeviceHelper reads `get_config().use_double` and sets float precision accordingly at `pysph/base/device_helper.py:47-77`.
- [UNKNOWN] It is not yet confirmed whether the first Warp implementation must support both float32 and float64 across all target GPUs.
