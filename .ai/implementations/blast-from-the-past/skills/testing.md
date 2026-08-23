# Skill: Testing

## When to Use

Use when selecting validation commands, benchmarks, or correctness checks.

## Commands

- Default tests: `python -m pytest -m "not slow" pysph`
- Make alias: `make test`
- Full tests: `python -m pytest pysph` or `make testall`
- Parallel/Zoltan tier: `python -m pytest -v -m 'slow or parallel'`

## Rules

- For performance claims, create an experiment entry with hardware, commit, command, inputs, repeated timings, and correctness checks.
- Compare Warp paths against an existing PySPH baseline before claiming speedup.
- Confirm with team: first concrete benchmark cases and acceptance thresholds.

## Required Closeout

Paste raw command output or benchmark numbers into the relevant experiment/review.
