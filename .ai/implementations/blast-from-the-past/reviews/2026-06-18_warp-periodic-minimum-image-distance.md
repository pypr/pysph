---
type: review
date: 2026-06-18
user: @kunalpuri-prediqt
agent: claude
plan: .ai/implementations/blast-from-the-past/plans/2026-06-18_warp-periodic-minimum-image-distance.md
adrs: [ADR-0004]
aspects_touched: [gpu-nnps, warp-backend, validation-benchmarks]
host_files: [pysph/base/warp_codegen.py, pysph/base/warp_nnps.py, pysph/base/warp_sph.py, pysph/base/tests/test_warp_codegen.py, pysph/base/tests/test_warp_sph.py]
status: approved
---

# Review - Periodic minimum-image neighbor distance (grid-direct path)

## Diff summary

- `pysph/base/warp_codegen.py`: `generate_group_source`/`build_group_kernel`
  gain `periodic=False` (grid only; in the cache key only when True, so
  non-periodic kernels keep their exact name/source/disk-cache). Periodic grid
  kernels add `box_lx/ly/lz` + `periodic_x/y/z` runtime int flags; the cell walk
  wraps the cell index per periodic dim (`((ix%nx)+nx)%nx`) else bounds-checks;
  `_emit_geometry` applies minimum image (`dx -= box_lx*wp.round(dx/box_lx)`) per
  periodic dim before `rij2`/cutoff.
- `pysph/base/warp_nnps.py`: `UniformGridWarpNNPS.set_periodic_box(bounds)`;
  `_compute_bounds_and_cell_size` tiles periodic dims (cubic box; `cell_size =
  L/floor(L/cell_min)`, requires `floor(L/cell_min) >= 3`) and stores box
  lengths + flags. **The cell-id binning kernels (`_cell_ids_counts_{f32,f64}`)
  now wrap (not clamp) the cell index in periodic dims**, so out-of-box source
  positions bin into their periodic image cell, consistent with the wrapped
  walk.
- `pysph/base/warp_sph.py`: `_run_equation_group` detects periodicity from
  `nnps._bounds` (grid mode) and builds the periodic variant + passes the box
  via `_grid_launch_args(periodic=True)`.
- Tests: codegen periodic (compiles, distinct, flat+periodic rejected); a
  CPU minimum-image parity test with an **out-of-box** source particle in the
  `nx=4` clamp-bug regime; a periodic-lattice uniformity (no boundary
  deficiency) test; a guard test (too-small box + missing min/max raise).

## Behavioral / numerical changes

- New capability: correct periodic neighbors (wrapped cell walk + minimum-image
  distance) for the grid-direct WCSPH paths -- enables periodic benchmarks
  (Taylor-Green etc.). The elliptical-drop (free-surface) path is unaffected.
- Non-periodic path byte-identical (periodic only engaged when a box is set;
  binning clamps when the dim is non-periodic). Continuity adaptive guard
  unchanged (`1393` steps, 0 recompiles -- disk cache intact).

## Tests / validation run

```text
$ python -m pytest -q test_warp_codegen.py test_warp_sph.py test_warp_nnps.py
57 passed, 2 warnings

periodic summation density vs CPU minimum-image (incl. out-of-box x=1.08, nx=4):
  max abs diff 5.51e-06; the out-of-box image is found by the in-box query.
periodic uniform lattice: std(rho) < 1e-4*mean, mean ~ rho0 (no boundary deficiency).
non-periodic regression: all prior tests pass; continuity guard 1393 steps, 0 recompiles.
```

## Adversarial pre-review (and fixes)

A 5-dimension workflow (min-image correctness, cell-walk wrap, NNPS tiling,
non-periodic preservation, launch binding), each finding skeptic-verified.
Result: **5 confirmed (1 major, 1 minor, 3 nits)** -- all addressed:

- **MAJOR (fixed)**: the source-particle binning kernel used `wp.clamp`, not a
  periodic wrap, so out-of-box source positions were mis-binned into the edge
  cell and the wrapped walk missed in-support periodic neighbors (reachable via
  the public API with un-wrapped positions, and the first `push=True` step
  before `wrap_periodic`). Fixed by wrapping the cell index in periodic dims in
  `_cell_ids_counts_{f32,f64}`; added the out-of-box regression test (verified
  it now matches the CPU min-image reference to 5.5e-06).
- **MINOR (fixed)**: a box narrower than `2*support` (support > L/2) was silently
  accepted via a `max(3, ...)` clamp, making minimum-image invalid. Replaced
  with a hard `floor(L/cell_min) >= 3` requirement that raises a clear error.
- **NIT (fixed)**: `periodic_in_x` without `xmin/xmax` raised a bare `KeyError`;
  now a clear `ValueError`. Guard test added.
- **NIT (fixed)**: the equal-length check compared post-floor effective cell
  sizes; now compares the raw periodic lengths.
- **NIT (no action)**: stored `_bounds` `ymin/zmin` differ from baseline for
  dims below `self.dim` -- dormant/non-behavioral (flagged for transparency).

## validate-memory.py

```text
validate-memory: PASS
```

## Risks

- MVP supports equal-length (cubic) periodic dims only; unequal lengths raise.
  Non-cubic periodic boxes / per-dim cell sizes are a follow-up.
- `wp.round` at exactly half-box is a measure-zero tie; bounded.
- fp32 visitation/min-image ~1e-6; within parity tolerances.

## Out of scope

- Ghost-particle approach; periodic support for the flat host-query path;
  non-cubic periodic boxes; cross-array periodic.

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > LGTM
