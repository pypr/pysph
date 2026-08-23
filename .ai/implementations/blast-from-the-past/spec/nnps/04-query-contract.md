# Neighbor Query Contract

## Single-Particle Query

The canonical public query is:

```text
get_nearest_particles(src_index, dst_index, d_idx, nbrs)
```

Inputs:

- source particle-array index
- destination particle-array index
- destination particle index
- mutable neighbor index array

Output:

- source-local particle indices appended into or written into `nbrs`

The result must contain source indices, not gids and not destination indices.

## Context

`set_context(src_index, dst_index)` prepares the implementation for repeated
queries between a source/destination pair. CPU implementations store current
source/destination wrappers and structure storage pointers.

Warp implementations should preserve this concept even if the actual kernels
receive source/destination arrays explicitly.

## Inclusion Rule

A source particle `j` is a neighbor of destination particle `i` when:

```text
norm(x_i - x_j, y_i - y_j, z_i - z_j) < radius_scale * h_i
or
norm(x_i - x_j, y_i - y_j, z_i - z_j) < radius_scale * h_j
```

Squared-distance comparisons are allowed and expected for performance, provided
the strict inequality is preserved.

## Ordering

Neighbor order is implementation-defined unless `sort_gids=True`.

When `sort_gids=True`:

- if source gids are valid, neighbors are sorted by source gid;
- if gids are invalid, neighbors are sorted by local source index.

Correctness tests should compare sets by default and ordered arrays only when
sorting is requested.

## Cache

With caching disabled, a query may compute neighbors immediately.

With caching enabled, an implementation may precompute:

- neighbor counts
- prefix sums
- flat neighbor lists

The cache must be invalidated after particle movement, structural mutation,
domain update, or spatial reordering.
