---
type: plan
id: 2026-06-18_warp-periodic-minimum-image-distance
author: @kunalpuri-prediqt
agent: claude
created: 2026-06-18T15:00:00 CEST
status: approved
aspects: [gpu-nnps, warp-backend, validation-benchmarks]
adr: ADR-0004
within_boundary: true
host_files:
  - pysph/base/warp_codegen.py
  - pysph/base/warp_nnps.py
  - pysph/base/warp_sph.py
  - pysph/base/tests/test_warp_codegen.py
  - pysph/base/tests/test_warp_sph.py
---

# Plan: Periodic minimum-image neighbor distance (grid-direct path)

## Goal

True periodic neighbors for the WCSPH device paths: find neighbors across
periodic boundaries (wrapped cell walk) and use minimum-image distances. Today
`wrap_periodic` only wraps particle *positions*; the cell walk has hard
`ix>=0 and ix<nx` bounds and `dx = d_x[i]-s_x[j]` is raw, so boundary particles
miss their wrap-around neighbors (boundary-deficiency artifacts). This blocks
correct periodic benchmarks (Taylor-Green vortex etc.). The elliptical drop is
free-surface and unaffected.

Approach: **minimum-image + wrapped cell walk**, not ghost particles -- the
natural fit for the uniform grid and the grid-direct generator. One generator
change covers continuity and summation paths (both are grid-direct).

## Approach

### 1. Codegen (`warp_codegen.py`) -- structural `periodic` flag

- `generate_group_source(..., periodic=False)` and `build_group_kernel(...,
  periodic=False)`, with `periodic` in `_cache_key`. **`periodic=False` is
  byte-identical to today** (non-periodic kernels do not change or recompile).
- `periodic=True` (grid mode only): add signature params after `radius_scale` --
  `box_lx, box_ly, box_lz` (typed) and `periodic_x, periodic_y, periodic_z`
  (`wp.int32` runtime flags, so one compiled variant serves any combination of
  periodic dimensions).
  - Cell walk: per dim, if the runtime flag is set wrap the index
    (`ix = (ix0+dxc+nx) % nx`, always in-bounds); else keep the bounds check.
  - Geometry pre-phase: after `dx = d_x[i]-s_x[j]`, apply minimum image when the
    dim flag is set (`dx -= box_lx * wp.round(dx/box_lx)`), before `rij2`/cutoff.

### 2. NNPS periodic box (`warp_nnps.py`)

- `UniformGridWarpNNPS.set_periodic_box(bounds)` stores the parsed periodic box.
  `_compute_bounds_and_cell_size` then, for periodic dims, uses the box extent
  and **tiles** it (`nx = max(3, floor(Lx/cell_size))`, effective
  `cell_size_x = Lx/nx`) so cell-index wrap is exact; non-periodic dims keep the
  particle-extent + pad behavior. `_bounds` carries `box_lx/ly/lz` and
  `periodic_x/y/z`.
- `_grid_launch_args` appends the box lengths + periodic int flags when the
  periodic variant is requested.

### 3. `warp_sph.py`

- `_run_equation_group` and the helpers gain `periodic=False`; when the NNPS has
  a periodic box, the WCSPH step paths build the periodic kernel variant and
  pass the box args. The step entry points already receive `periodic_bounds`;
  they set it on the NNPS (`set_periodic_box`) so the grid tiles and the launch
  carries the box.

## Tests / validation

- `test_warp_codegen.py`: a `periodic=True` grid kernel compiles, caches
  distinctly from `periodic=False`, and on a small 1D-periodic fixture a
  wrap-around neighbor is found with the minimum-image distance (vs a hand
  calculation).
- `test_warp_sph.py`: summation density / pressure gradient on a small periodic
  box match a brute-force **CPU minimum-image** reference (fp32 tol); a periodic
  uniform-lattice **symmetry** test (every particle has the same neighbor
  count/density, interior == boundary).
- Non-periodic regression: the existing 53 tests are unchanged (periodic=False
  path byte-identical); continuity adaptive `nx=100` guard still `1393` steps.
- Adversarial-review workflow; @prabhu sign-off.

## Success criteria

- Periodic grid-direct neighbors match a CPU minimum-image reference at fp32
  scale; periodic-lattice density is uniform (no boundary deficiency); the
  non-periodic path is provably unchanged (no recompile, suite green).

## Risks

- Requires `nx >= 3` cells per periodic dim and `cell_size >= radius_scale*h`
  with box >= 2*support (standard SPH periodic constraint); guard with a clear
  error.
- `wp.round` at exactly half-box is a measure-zero tie; bounded.
- The periodic variant adds per-dim runtime branches; only compiled/used when a
  periodic box is set, so the validated non-periodic path is untouched.

## Out of scope

- Ghost-particle approach; periodic support for the flat
  `build_neighbor_cache_gpu` host-query path (device paths are all grid-direct);
  mixed periodic/mirror domains; cross-array periodic.

## Approval

- [x] Plan posted in chat; purpose clarified; direction confirmed
- Approved by: @kunalpuri-prediqt at 2026-06-18T15:00:00 CEST
- Approval, verbatim quote (AskUserQuestion selection):
  > Do (2) then prepare (1)
