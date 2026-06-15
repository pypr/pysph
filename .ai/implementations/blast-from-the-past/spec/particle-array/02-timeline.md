# 02 Timeline

## Lifecycle Stages

1. [OBSERVED] Construction starts from user arrays or property descriptor dictionaries passed to `ParticleArray` or `get_particle_array()`; scalar values are accepted and broadcast by tests at `pysph/base/tests/test_particle_array.py:1020-1030`.
2. [OBSERVED] `get_particle_array()` fills default descriptors for common properties, chooses int/uint/double types, sets `gid` default to `UINT_MAX`, and marks selected output arrays at `pysph/base/utils.py:47-149`.
3. [OBSERVED] `ParticleArray._initialize()` clears existing state, derives particle count from the first/maximum property size, ravels input data, broadcasts scalar data to the inferred count, calls `add_property()`, and aligns particles at `pysph/base/particle_array.pyx:225-293`.
4. [OBSERVED] For non-cython backends, the constructor attaches `DeviceHelper`, which creates device arrays for all current properties and constants at `pysph/base/particle_array.pyx:149-157` and `pysph/base/device_helper.py:56-70`.
5. [OBSERVED] Runtime mutation can add/remove/extend/resize/append/extract/copy properties and particles through ParticleArray methods declared at `pysph/base/particle_array.pxd:75-138`.
6. [OBSERVED] Alignment partitions Local-tagged particles first and updates `num_real_particles` at `pysph/base/particle_array.pyx:1092-1173`.
7. [OBSERVED] Device-backed ParticleArray methods delegate selected operations to `DeviceHelper` when `self.gpu is not None and self.backend is not 'cython'`, for example remove/add/extract/align/resize at `pysph/base/particle_array.pyx:439-505`, `pysph/base/particle_array.pyx:531-602`, `pysph/base/particle_array.pyx:1237-1277`, `pysph/base/particle_array.pyx:1092-1131`, and `pysph/base/particle_array.pyx:1438-1445`.
8. [OBSERVED] Output obtains metadata with `get_particles_info()` and property arrays with `get_property_arrays()` before writing, and `get_property_arrays()` pulls device data first for non-cython backends at `pysph/base/utils.py:466-497`, `pysph/solver/output.py:53-78`, and `pysph/base/particle_array.pyx:344-386`.
9. [OBSERVED] Load/restart reconstructs ParticleArray objects from saved property metadata and arrays in NumPy/HDF output loaders at `pysph/solver/output.py:127-162` and `pysph/solver/output.py:195-221`.

## Representative Single-Step Context

- [OBSERVED] NNPS update is called after particles move, assumes each processor already has needed local particle information in parallel runs, computes/refreshes local data structures, and bins all particles reported by ParticleArray at `pysph/base/nnps_base.pyx:1471-1506`.
- [INFERRED] In a solver time step, ParticleArray provides mutable property arrays before and after equations/integrators move particles; the numerical update itself is outside this layer.
- [OBSERVED] Periodic/mirror domain update removes old Ghost-tagged particles and can create new ghost images before binning at `pysph/base/nnps_base.pyx:386-403` and `pysph/base/nnps_base.pyx:450-470`.

## Sync Timeline

- [OBSERVED] `DeviceHelper.push()` copies selected or all host arrays to device, and tests cover both selective and full push at `pysph/base/device_helper.py:219-227`, `pysph/base/tests/test_device_helper.py:51-72`, and `pysph/base/tests/test_device_helper.py:74-95`.
- [OBSERVED] `DeviceHelper.pull()` copies selected or all device arrays back to host and synchronizes `num_real_particles`, with tests at `pysph/base/device_helper.py:200-217`, `pysph/base/tests/test_device_helper.py:97-118`, and `pysph/base/tests/test_device_helper.py:120-141`.
- [INFERRED] A Warp migration must define when host data become stale and whether `get()`, attribute access, output, and Cython wrapper reads trigger implicit synchronization.
