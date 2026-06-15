# Open Questions - particle-memory

- [open] Should Warp arrays mirror existing device helpers or become a separate backend-owned representation?
- [open] What host-device synchronization points are required for current PySPH outputs?
- [open] Should constants be Warp arrays, host-only metadata, or mirrored both ways?
- [open] What exact post-mutation ordering guarantees should the Warp backend promise for strided/tagged arrays?
