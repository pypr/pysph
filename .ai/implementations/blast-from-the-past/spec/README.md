# Specification Index

This directory contains implementation-scoped specifications for
`blast-from-the-past`.

## Codebase-Level Context

- `codebase-understanding/`: curated implementation-facing digest of the
  top-level `CODEBASE_UNDERSTANDING.md` report.

## Migration Targets

- `particle-array/`: solver-agnostic specification for ParticleArray storage,
  mutation, host/device synchronization, and the first Warp device mirror.
- `nnps/`: solver-agnostic specification for nearest-neighbor particle search,
  domain updates, caching, and the next Warp migration layer.

## Reading Order

1. `codebase-understanding/00-overview.md`
2. `codebase-understanding/01-runtime-flow.md`
3. `particle-array/00-overview.md`
4. `nnps/00-overview.md`
5. `codebase-understanding/03-gpu-migration-map.md`
