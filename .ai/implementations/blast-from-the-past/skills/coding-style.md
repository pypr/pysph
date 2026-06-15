# Skill: Coding Style

## When to Use

Use when planning or editing host Cython/Python code for `blast-from-the-past`.

## Rules

- Honor existing PySPH Cython style and public surfaces.
- Keep `.pxd` declarations and `.pyx` implementations synchronized.
- Do not introduce new format/lint tooling without an ADR.
- Use clear names for benchmarks and Warp prototypes; avoid encoding performance claims in names.
- Add comments only where GPU/Warp/Cython ownership or lifetime is not obvious.

## Required Closeout

Record host files touched, tests run, and any style deviations in the session log.
