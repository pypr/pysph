# Codebase Understanding Overview

## Source

This spec layer distills the repository-level `CODEBASE_UNDERSTANDING.md`
document into implementation-scoped guidance for `blast-from-the-past`.

The source document describes PySPH as a general-purpose SPH framework with
performance-sensitive code generated or compiled through Cython, OpenCL, CUDA,
OpenMP, MPI, and optional Zoltan paths.

## High-Level Product Shape

PySPH is not a single solver binary. It is a framework where users:

1. create particle arrays,
2. select or write schemes and equations,
3. let `Application` build solver, domain, NNPS, integrator, and compiler
   objects,
4. run a timestep loop,
5. write particle-output files for visualization and post-processing.

The GPU migration must therefore preserve framework extensibility. The target
is not one CUDA-only solver path; it is a backend path that can eventually serve
many schemes.

## Core Subsystems

The source understanding identifies five core runtime subsystems:

- `pysph/base`: ParticleArray, typed arrays, kernels, domain managers, NNPS, GPU
  NNPS exports.
- `pysph/sph`: Equation abstraction, schemes, integrators, backend code
  generation, compiler helpers.
- `pysph/solver`: Application lifecycle, solver loop, output, callbacks, command
  orchestration.
- `pysph/parallel`: MPI/Zoltan particle exchange, load balancing, and remote
  particle management.
- `pysph/tools`: CLI, examples, post-processing, VTK/XDMF utilities.

## Migration Implication

The GPU path must be staged:

1. Particle storage/mirror semantics.
2. Domain and NNPS on device.
3. Equation and integrator kernel consumption of device data.
4. Solver/Application selection and end-to-end examples.
5. Parallel and output correctness.

Skipping directly to equation kernels would leave neighbor search and domain
updates as synchronization bottlenecks.
