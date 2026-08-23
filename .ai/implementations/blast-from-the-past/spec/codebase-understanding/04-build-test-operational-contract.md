# Build, Test, And Operational Contract

## Build Inputs

The source understanding identifies a hybrid build:

- Python packaging through `pyproject.toml` and `setup.py`.
- Cython extensions for low-level arrays, NNPS, tools, and parallel paths.
- Optional OpenMP detection.
- Optional MPI/Zoltan detection.
- Local build overrides through `~/.compyle/config.py`.

For the PQT environment, PySPH is installed editable and Zoltan is available at:

```text
/home/kunalp/prediqt/zoltan
```

## Runtime Backend Controls

Existing Application options include:

- OpenMP controls,
- OpenCL/CUDA selection,
- precision selection,
- kernel choice,
- NNPS choice,
- parallel/Zoltan controls,
- output controls.

Warp integration should follow this pattern by making backend selection explicit
and discoverable rather than hidden inside one data structure.

## Test Gates

The source document identifies:

- default local tests excluding `slow`,
- full test target for all tests,
- separate MPI/Zoltan workflows,
- parallel tests that compare serial and parallel outputs by final time and
  particle coordinates keyed by global id.

Implementation-specific gates:

- ParticleArray Warp tests: `pysph/base/tests/test_warp_device_helper.py`.
- ParticleArray experiment wrapper:
  `experiments/2026-06-15_initial-warp-benchmark-placeholder/run_correctness.sh`.
- Future NNPS gates should include CPU-vs-Warp neighbor fixtures before solver
  integration.

## Operational Notes

Generated code and compiler caches live outside the repository, notably under
`~/.pysph` and `~/.compyle`. Reproducible GPU experiments should record:

- Python environment,
- Warp version,
- GPU device,
- precision mode,
- compiler/cache state when relevant,
- backend/NNPS selection,
- particle count and average neighbor count.
