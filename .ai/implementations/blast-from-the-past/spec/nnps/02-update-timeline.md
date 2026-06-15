# Update Timeline

## Normal Serial Update

The observed CPU `NNPS.update()` sequence is:

1. Read `cell_size` and `hmin` from the domain manager.
2. Compute coordinate bounds across particle arrays.
3. Refresh structure storage.
4. For each particle array, create an index list for all particles.
5. Bin local particles into the structure.
6. Rebuild neighbor caches when caching is enabled.

The observed GPU `GPUNNPS.update()` sequence is similar but lets concrete GPU
subclasses perform device-side `_bin()` and `_refresh()`.

## Domain Update

`update_domain()` calls `domain.update()`. Domain update may:

- remove old ghost particles
- recompute cell size
- create periodic or mirror ghosts
- update local domain state

NNPS must be valid only after the domain update and structure update have both
run for the current particle positions.

## Solver Loop Placement

From the high-level solver flow, NNPS update participates in:

- initial setup before acceleration computation
- post-stage domain updates during integrator stages
- optional spatial reordering
- parallel manager exchange and load-balance updates

Warp NNPS integration should initially target explicit `nnps.update()` and
query calls before entering full solver-loop orchestration.
