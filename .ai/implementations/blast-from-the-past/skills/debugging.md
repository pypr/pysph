# Skill: Debugging

## When to Use

Use when diagnosing GPU, Cython, memory, or benchmark failures.

## Rules

- First preserve the failing command, inputs, environment, and exact error.
- Separate build failures, code-generation failures, runtime GPU failures, and numerical mismatches.
- For GPU memory issues, record ownership and transfer assumptions in `particle-memory`.
- For NNPS mismatches, record destination/source array, particle counts, dtype, and neighbor count differences.

## Required Closeout

Log the failure mode, suspected layer, evidence, and next concrete reproduction command.
