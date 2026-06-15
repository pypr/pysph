# Skill: Working with Host Code

## When to Use

Use before reading or modifying host files.

## Rules

- Stay inside the integration boundary in `.ai/implementations/blast-from-the-past/implementation.md`.
- Do not catalogue the host beyond what the task needs.
- Propose an ADR before changing public Cython ABI/API surfaces.
- Do not copy secrets into `.ai/`; reference secret locations by path only.
- If a plan touches outside-boundary files, set `within_boundary: false` and require a boundary amendment at review time.

## Required Closeout

Record host files consulted or changed and whether the boundary remained truthful.
