# Subsystem Map

## Particle Data

`ParticleArray` is the central mutable state container. It stores named
properties, constants, output-array metadata, stride metadata, and optional GPU
helpers. Standard properties include position, velocity, mass, smoothing length,
density, pressure, acceleration, global id, process id, and particle tag.

Implementation specs:

- `../particle-array/`

## Domain And Neighbor Search

The domain manager owns physical bounds, periodic/mirror flags, ghost layers,
cell size, and smoothing-length-derived binning state. NNPS builds the local
neighbor-search structure and answers source/destination neighbor queries.

Implementation specs:

- `../nnps/`

## Equation/Scheme Layer

Equations define operations over destination and source particle arrays.
Schemes assemble common equation groups, kernels, integrators, solver options,
and required particle properties.

Warp migration consequence:

- Equation signatures and property names are compile-time contracts.
- Missing particle properties fail during acceleration-evaluator setup.
- Device data layout must preserve the flat property arrays expected by
  generated equation code.

## Solver/Application Layer

`Application` wires user options, particles, schemes, solver, domain, NNPS,
parallel manager, tools, callbacks, and output. It is the likely host surface
for future `warp` backend selection.

## Parallel Layer

Parallel execution depends on MPI and Zoltan. The parallel manager removes stale
remote particles, repartitions or migrates local particles, imports remote
particles, and updates local/remote cell information before local computation.

Warp migration consequence:

- NNPS and ParticleArray must tolerate local/remote/ghost ordering changes.
- Parallel correctness should compare by `gid` where ordering differs.

## Output Layer

Solver output serializes particle arrays and solver metadata to NPZ or HDF5,
with optional VTK/XDMF conversion. NNPS structures and generated kernels are not
serialized as solver output.

Warp migration consequence:

- Device data must be explicitly pulled before output paths read host arrays.
- Output should remain backend-neutral.
