# GPU Migration Map

## Existing Accelerator Model

The source document describes PySPH GPU acceleration as generated-code based:

- Python/Mako helper code generates OpenCL/CUDA evaluator and integrator code.
- `DeviceHelper` mirrors ParticleArray data into Compyle arrays.
- Existing GPU NNPS classes are exported from `pysph/base/gpu_nnps.py`.
- Application setup selects GPU NNPS when OpenCL or CUDA runtime flags are set.

The repository does not contain handwritten CUDA C kernels as the primary GPU
model.

## Warp Positioning

Warp should be introduced as an additive backend path, not as a replacement for
existing OpenCL/CUDA/Compyle behavior.

Current staged position:

1. `backend="warp"` ParticleArray mirror exists as a prototype.
2. NNPS solver-agnostic spec exists.
3. Warp NNPS implementation is the next proposed migration layer.

## Recommended Order

1. **ParticleArray mirror:** preserve host API, prove dtype/stride/constants,
   push/pull, add/remove/extract/append, and alignment.
2. **NNPS correctness:** compare Warp neighbor sets against CPU baselines.
3. **NNPS performance:** implement device-side cell-list or equivalent
   structure and separate update/query/cache/readback timing.
4. **Application selection:** expose explicit Warp NNPS/backend flags after the
   direct API is stable.
5. **Equation consumption:** decide whether Warp equation kernels consume cached
   neighbor lists or launch query kernels directly.
6. **End-to-end examples:** run small examples through solver setup and output.
7. **Parallel verification:** test Zoltan/MPI exchange plus Warp update/query.

## Avoided Shortcut

Do not migrate equation kernels before NNPS. The codebase understanding makes
neighbor search part of the hottest solver path, and a CPU NNPS would force
position and neighbor synchronization every timestep.
