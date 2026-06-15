# 00 Overview - Particle Array

## Scope

- [OBSERVED] `ParticleArray` is the main PySPH data structure representing a homogeneous collection of particles with arbitrary named properties stored as `BaseArray` instances; this is described in the user guide at `docs/source/using_pysph.rst:59-76`.
- [OBSERVED] This spec focuses on `pysph/base/particle_array.pxd`, `pysph/base/particle_array.pyx`, `pysph/base/device_helper.py`, and construction/serialization helpers in `pysph/base/utils.py`; these files define the class, host storage, device mirroring, and metadata reconstruction paths at `pysph/base/particle_array.pxd:37-138`, `pysph/base/particle_array.pyx:109-157`, `pysph/base/device_helper.py:47-70`, and `pysph/base/utils.py:466-512`.
- [INFERRED] The solver-agnostic object here is not a numerical solver, flux, equation, or integrator; it is a mutable particle table plus host/device synchronization protocol used by solvers and neighbor search.
- [OBSERVED] PySPH's default particle properties include positions, velocities, smoothing length, mass, density, pressure, acceleration-like fields, `gid`, `pid`, and `tag`; their documented types are double for most physical fields, unsigned int for `gid`, and int for `pid`/`tag` at `docs/source/using_pysph.rst:78-89`.

## Existing Role

- [OBSERVED] `ParticleArray` owns dictionaries for `properties`, `constants`, `default_values`, `stride`, `output_property_arrays`, `lb_props`, plus `backend`, `gpu`, `time`, and `num_real_particles`; this is declared in the Cython header at `pysph/base/particle_array.pxd:37-73`.
- [OBSERVED] The constructor resolves a backend, initializes host properties, adds constants, configures load-balance/output metadata, and attaches `DeviceHelper` when the backend is not `cython` at `pysph/base/particle_array.pyx:109-157`.
- [OBSERVED] `DeviceHelper` manages device-side mirrors for ParticleArray properties/constants and exposes push/pull/update operations at `pysph/base/device_helper.py:47-70` and `pysph/base/device_helper.py:200-227`.
- [OBSERVED] Solver output obtains particle metadata and property arrays through `get_particles_info()` and `get_property_arrays()` before dumping data at `pysph/base/utils.py:466-497` and `pysph/solver/output.py:53-78`.
- [OBSERVED] NNPS update bins all particles reported by `pa.get_number_of_particles()` after particle movement and domain refresh at `pysph/base/nnps_base.pyx:1471-1506`.

## Non-Goals For This Spec

- [INFERRED] This spec does not define SPH governing equations because ParticleArray does not implement numerical residuals or time integration.
- [INFERRED] This spec does not define NNPS algorithms beyond the ParticleArray contract they consume.
- [INFERRED] This spec does not prescribe a final Warp architecture; it identifies the contract that a Warp migration must preserve.

## Representative Call Graph

- [OBSERVED] Application creation path: `Application._create_particles()` creates or loads particles, records metadata with `utils.get_particles_info()`, broadcasts metadata in MPI runs, and creates dummy arrays on non-root ranks at `pysph/solver/application.py:859-920`.
- [OBSERVED] User/helper creation path: `get_particle_array()` builds default property descriptors, merges user properties, chooses property dtypes/defaults, constructs `ParticleArray`, and sets output arrays at `pysph/base/utils.py:47-149`.
- [OBSERVED] ParticleArray initialization path: `ParticleArray.__init__()` calls `_initialize()`, which computes the particle count, ravels/broadcasts input data, adds properties, and calls `align_particles()` at `pysph/base/particle_array.pyx:109-157` and `pysph/base/particle_array.pyx:225-293`.
- [OBSERVED] Device path: for non-cython backends, `ParticleArray` creates `DeviceHelper`, which materializes device arrays for properties/constants and tracks `num_real_particles` at `pysph/base/particle_array.pyx:149-157` and `pysph/base/device_helper.py:56-70`.
- [OBSERVED] Output path: output code calls `get_particles_info()` and `get_property_arrays()`, and `get_property_arrays()` pulls requested GPU arrays first when the backend is not CPU-only at `pysph/solver/output.py:53-78` and `pysph/base/particle_array.pyx:344-386`.

## Minimal Invariants

- [OBSERVED] A property is a one-dimensional array; strided properties are represented by a flat one-dimensional array plus a `stride` entry at `docs/source/using_pysph.rst:94-104`.
- [OBSERVED] `tag`, `pid`, and `gid` are baseline properties after `clear()` at `pysph/base/particle_array.pyx:395-400`.
- [OBSERVED] `align_particles()` moves Local-tagged particles to the start and updates `num_real_particles` at `pysph/base/particle_array.pyx:1092-1173`.
- [OBSERVED] Constants are separate from particle properties and do not resize when particles are added at `docs/source/using_pysph.rst:148-175` and `pysph/base/tests/test_particle_array.py:805-817`.
