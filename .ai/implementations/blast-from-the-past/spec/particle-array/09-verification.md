# 09 Verification

## Baseline Test Set

- [OBSERVED] `pysph/base/tests/test_particle_array.py` is the main behavior suite for ParticleArray construction, property access, mutation, constants, serialization helpers, CPU behavior, and OpenCL/CUDA variants at `pysph/base/tests/test_particle_array.py:28-1139`.
- [OBSERVED] `pysph/base/tests/test_device_helper.py` is the main behavior suite for backend mirror synchronization and device-side mutation operations at `pysph/base/tests/test_device_helper.py:29-455`.
- [OBSERVED] NNPS tests create ParticleArray instances through `get_particle_array()` and depend on position/smoothing-length semantics at `pysph/base/tests/test_nnps.py:12-83`.
- [INFERRED] The first Warp migration should run ParticleArray and DeviceHelper-equivalent tests before broader solver/NNPS tests.

## Focused Acceptance Matrix

- [INFERRED] Construction: raw arrays, dict descriptors, scalar broadcast, flattened 2D data, default properties, and constants.
- [INFERRED] Mutations: add/remove particles, remove tagged particles, append arrays, extract particles, resize, extend, add/remove properties, and copy properties.
- [INFERRED] Stride: all mutation and alignment operations must be tested with stride greater than 1.
- [INFERRED] Sync: selective/full push and pull must be tested with both float and integer properties.
- [INFERRED] Serialization: `get_particles_info()`, `create_dummy_particles()`, pickle, NumPy output, and HDF output metadata should remain compatible.
- [INFERRED] Boundary-adjacent behavior: Ghost-tag removal and periodic/mirror append/extract paths should be covered by NNPS/domain tests after the storage layer passes.

## Spec-Derived Warp Smoke Tests

- [INFERRED] Create a Warp-backed ParticleArray with `x`, `y`, `z`, `h`, `m`, `rho`, `tag`, `pid`, `gid`, and one stride-3 property.
- [INFERRED] Mutate host `x` and `tag`, push selectively, align on device, pull `x/tag`, and assert Local particles are first with strided values reordered consistently.
- [INFERRED] Add particles with a missing property and verify defaults are filled on device and host after pull.
- [INFERRED] Remove Ghost-tagged particles on device and verify count, tag partition, and stride data.
- [INFERRED] Dump and reload metadata without requiring Warp to be present for the loaded host arrays.

## Validation Commands

- [OBSERVED] The memory system validator is available at `.ai/implementations/blast-from-the-past/scripts/validate-memory.py`, whose script header and root discovery are at `.ai/implementations/blast-from-the-past/scripts/validate-memory.py:1-12`.
- [INFERRED] Phase 1 spec validation should run `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py` and `git diff --check -- .ai AGENTS.md`.
- [INFERRED] Phase 2 runtime probing should use the user-provided environment activation command before importing Warp: `source $HOME/prediqt/activate && python -c "import warp; print(warp.__version__)"`.
- [UNKNOWN] The exact PySPH test command for Warp-specific tests is not defined until the Warp test module path exists.
