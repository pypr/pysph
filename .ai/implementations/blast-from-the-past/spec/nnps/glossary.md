# Glossary

| Term | Meaning |
| --- | --- |
| NNPS | Nearest-neighbor particle search. |
| Source array | Particle array from which neighbor indices are returned. |
| Destination array | Particle array containing the queried particle. |
| `d_idx` | Destination-local particle index. |
| Neighbor index | Source-local particle index satisfying the interaction radius test. |
| Radius scale | Kernel support multiplier applied to smoothing length `h`. |
| Cell size | Spatial bin size used by cell/hash/list NNPS variants. |
| Cache | Precomputed neighbor lengths, starts, and flat neighbor indices. |
| Ghost particle | Particle copied or synthesized for periodic, mirror, or parallel overlap. |
| Remote particle | Particle imported from another MPI rank for local interaction. |
| Spatial reordering | Alignment of particle properties by an NNPS-provided locality permutation. |
