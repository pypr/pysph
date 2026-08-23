# Skill: Long-Running Tasks

## When to Use

Use for tasks over 30 minutes, over 200 LOC, over five files, or any numerical run that must complete.

## Rules

- Do not present placeholders as completed work.
- Long runs are tracked as experiments with `status: running`.
- Every boot checks running experiments first and updates `last_checked`.
- If context ends, write exact file:line, test/experiment state, and next action.

## Required Closeout

Update the experiment and `current.md` with the next checkpoint.
