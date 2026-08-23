# Domain And Geometry

## Domain State

The domain manager owns:

- physical limits: `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`
- periodic flags per axis
- mirror flags per axis
- ghost-layer count
- cell size
- minimum smoothing length
- radius scale
- whether the run is in parallel
- particle-array wrappers and copy-property selection for ghosts

## Bounds Update

NNPS bounds are computed from particle coordinates. The observed CPU and GPU
paths expand min/max bounds by one percent of the current coordinate extent and
fall back to a unit-sized box when all extents are near zero.

The Warp implementation should preserve these semantics so that CPU/Warp
neighbor queries agree in degenerate and small-domain cases.

## Cell Size

The domain manager computes the binning cell size from smoothing-length state
and radius scale. NNPS consumes `domain.manager.cell_size` and
`domain.manager.hmin` during update.

For fixed smoothing length cases, the cell structure may be reused more
aggressively, but the externally visible result must not depend on that
optimization.

## Dimensionality

The NNPS constructor receives `dim`, but position storage always uses `x`, `y`,
and `z`. A backend may ignore unused axes when computing flattened cell ids and
valid neighbor cell shifts.

Required dimensions:

- 1D: x only
- 2D: x, y
- 3D: x, y, z
