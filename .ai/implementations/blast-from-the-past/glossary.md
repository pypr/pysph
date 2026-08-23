# Glossary

- **blast-from-the-past** - Implementation track exploring NVIDIA Warp for PySPH GPU particle dynamics.
- **Warp** - NVIDIA Warp, the candidate GPU programming/runtime layer for this implementation. Confirm exact package version and docs before ADRs depend on API details.
- **NNPS** - Nearest-neighbor particle search; a core PySPH hot path and major GPU integration target.
- **ParticleArray** - PySPH particle storage abstraction with typed properties and optional device helpers.
- **Device helper** - Existing PySPH/Compyle bridge that mirrors particle arrays to GPU/device arrays.
- **Boundary** - The approved set of host files/modules this implementation may touch.
- **Experiment** - A tracked benchmark, validation run, or parameter study whose result may inform an ADR.
