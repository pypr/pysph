---
type: experiment
id: 2026-07-06_warp-adaptive-particle-resolution-p0
created: 2026-07-06T10:45:00 CEST
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: active
last_checked: 2026-07-06T10:47:48 CEST
---

# Experiment: Warp adaptive particle resolution P0

## Purpose

Establish the non-adaptive fixed-obstacle baseline and select evidence-backed
3D split/merge primitives before designing the multilevel NNPS or device pool.

## Setup

- Existing `WarpDamBreak3DRunner`, extended additively with
  `--with-obstacle`; fluid index 0, wall index 1, obstacle index 2.
- Local active Warp environment and GPU recorded with every output.
- Candidate split patterns drawn from Vacondio et al. 2016 and the audited
  Muta--Ramachandran source, evaluated first with NumPy density reconstruction.

## Hypothesis

The existing multi-solid WCSPH driver will run the fixed Kleefsman obstacle
without backend changes. The published icosahedral/central 3D split should have
lower density error than a cubic eight-child split, but its particle-growth
cost may justify an iterative split/merge alternative on the GPU.

## Execution

1. Run a coarse 1--5 step obstacle smoke case and save metrics/state.
2. Run a longer coarse transient only after the smoke remains finite.
3. Ingest exact published 3D stencil parameters and reproduce density-error
   curves in a standalone kill test.
4. Compare candidate allocation count, neighbor count, isotropy, conservation,
   and density error in fp32/fp64.

## Results

### Reference/source audit

- The open Muta--Ramachandran PySPH implementation confirms the desired
  parallel workflow, but its automatic background path is restricted to 2D,
  several target-mass equations have explicit 3D FIXMEs, and its GPU particle
  lifecycle raises `NotImplementedError`. It is a process reference, not a 3D
  implementation to transplant.
- Vacondio et al. compare four 3D stencils and select a 12-vertex icosahedron
  plus center (13 daughters) with Wendland `epsilon=0.65`, `alpha=0.70`.

### Fixed-obstacle runner checkpoint

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB, `sm_89`; Warp 1.14.0,
CUDA toolkit 12.9, driver API reported as 13.2; fp32 backend.

At `dx=0.10`, the shared `DamBreak3DGeometry` creates 1,000 fluid, 3,824 wall,
and 4 fixed obstacle particles. The first obstacle execution paid the known
generated-Wendland cache load (three generated modules reported 40.410,
54.966, and 52.890 seconds). Warm results:

```text
case                       steps   t             elapsed   finite
obstacle smoke                 1   0.000001171   cached    true
obstacle startup              20   0.003095018   3.16 s    true
obstacle first impact        250   0.258454926   5.76 s    true
```

First-impact metrics:

```text
surge_front_x:             2.4901464 m
max_height:                0.9764588 m
fluid rho:                 983.44897 .. 1022.19061 kg/m^3
fluid p:                   -16.946 .. 25.564 kPa
wall p:                    0 .. 49.672 kPa
obstacle p:                23.696 .. 150.147 kPa
kinetic_energy:            292.19788 J
device obstacle drift:     0 exactly (step 1 vs step 250 arrays)
process max RSS:           350,756 KiB
```

The no-obstacle one-step compatibility run also remains finite with the original
1,000 fluid + 3,824 wall arrays and reports `obstacle_particles=0`.

### Split-stencil density kill test

`split_stencil_density_kill.py` implements the paper's constrained global
density-error minimization with the exact PySPH/Warp 3D WendlandQuintic C2
formula. A deterministic tensor integration converges at 81/101/121 points per
axis. At 101 points and the paper's `epsilon=0.65`, `alpha=0.70`:

```text
stencil                 daughters   optimized E       equal-mass E
cubic + center                   9   1.2586503e-3      1.2687077e-3
icosahedron + center            13   3.5803204e-4      5.5872945e-4
```

The icosahedral constrained masses are 12 x `0.0739476671` on the shell plus
`0.1126279952` at the center; mass sums to exactly 1 in the calculation. This
confirms the icosahedral advantage and the value of unequal masses. However,
the reproduction does **not** match the paper's Table 1 value (`E=8.326e-5`,
`min/max mass=0.33`): the current calculation gives `E=3.58032e-4` and
`min/max=0.656566`. Grid convergence rules out quadrature resolution as the
cause. Kernel/smoothing-length convention or an unrecorded stencil detail must
be resolved before the published masses are treated as an oracle.

## Conclusion

The existing multi-solid Warp backend can run the fixed Kleefsman obstacle
without backend changes. This resolves the obstacle-wiring part of P0. P0
remains active: exact 3D split constants, density-error kill tests, probes, and
uniform coarse/fine baselines are still required before ADR-0007. The first
density kill test correctly blocks the ADR because it exposes a paper/PySPH
convention mismatch.

## Follow-ups

- Resolve the Vacondio/PySPH Wendland or stencil convention mismatch exposed by
  the converged density-error reproduction.
- Add obstacle probe/impulse metrics and uniform coarse/fine baselines.
- Select ADR-0007 only after scientific and GPU-cost evidence agree.
