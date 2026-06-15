# Runtime Flow

## Application Flow

The source report identifies the main runtime path:

```text
pysph console script
-> pysph.tools.cli.main
-> pysph run or direct example execution
-> Application.run()
-> Application.setup()
-> create particles, scheme, equations, solver, domain, NNPS
-> Solver.setup()
-> SPHCompiler.compile()
-> Solver.solve()
```

## Solver Loop

The solver loop performs:

1. initial output and optional spatial reorder,
2. initial acceleration computation,
3. timestep computation,
4. pre-step callbacks,
5. integrator stage execution,
6. post-stage or post-step callbacks,
7. domain/NNPS updates as needed,
8. output/progress/commands,
9. final output.

## Generated Code Boundary

SPH equations are authored in Python classes, normalized into equation groups,
and converted to backend-specific generated code. CPU paths use Cython.
Existing GPU paths use OpenCL/CUDA helpers and Mako templates through Compyle.

For Warp migration, this means the immediate public API is not just an array
object. The eventual target is generated or staged computation that consumes
ParticleArray and NNPS state without excessive host synchronization.

## Immediate Integration Boundary

Until equation-kernel migration begins, Warp work should expose compatibility
through existing host-facing contracts:

- `ParticleArray` methods and properties,
- `NNPS.update()` and `get_nearest_particles()`,
- explicit backend or NNPS selection in Application setup,
- output paths that can pull host-readable arrays when requested.
