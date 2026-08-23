# Conventions

## Inherited from Host

- Use existing PySPH build and test commands before introducing new tooling.
- Keep Cython declarations and implementations consistent across `.pxd` and `.pyx` files.
- Preserve existing public names and import surfaces unless an ADR explicitly approves a change.
- Treat slow and parallel tests as separate validation tiers.

## Specific to blast-from-the-past

- Keep Warp exploration behind explicit decisions until a stable integration path exists.
- Record performance claims as experiments with hardware, command, inputs, output numbers, and correctness checks.
- Do not call a benchmark "fast" without a baseline and repeated measurement.
- Use `Confirm with team` where Prabhu guidance or hardware assumptions are not yet documented.
- Never copy credentials or private machine paths that expose secrets into `.ai/`.
