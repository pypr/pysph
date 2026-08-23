---
type: review
date: 2026-07-06
user: @kunalpuri-prediqt
agent: codex
plan: 2026-07-06_warp-multilevel-gpu-nnps
adrs: [ADR-0003, ADR-0004]
aspects_touched: [gpu-nnps, warp-backend, particle-memory, validation-benchmarks]
host_files: []
review_mode: prototype-owner
status: prototype-approved
---

# Review - multilevel GPU NNPS plan checkpoint

## Diff summary

- Add a focused Tier-2 draft plan for an exact, device-built multilevel Warp
  NNPS as the first runtime GPU APR milestone.
- Bound the proposed implementation away from split/merge allocation, APR
  physics, offline stencil optimization, periodic traversal, and multi-GPU.
- Define correctness, device-residency, candidate-count, runtime, and memory
  acceptance gates before ADR-0007 may be accepted.
- Refresh active-experiment inspection timestamps and current/daily/session
  memory after checking that recorded artifacts remain present.

## Aspects touched and host files modified

Planning touches `gpu-nnps`, `warp-backend`, `particle-memory`, and
`validation-benchmarks`. No host PySPH file is modified in this checkpoint.

## Behavioral / numerical changes

None. This checkpoint records a draft plan and memory only. It does not change
the NNPS, generated Warp kernels, simulation behavior, or numerical results.

## Tests / validation run

```text
$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS

$ git diff --check
<no output; exit 0>
```

No numerical test was run because no executable or host code changed.

## validate-memory.py

```text
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: n-a
- Amendments log entry: n-a

Only `.ai/` plan and implementation-memory files changed.

## Visual aid

Waiver: this is a plan-only checkpoint with no runtime or scientific result to
visualize.

## Risks

- The plan is still `status: draft`: committing it records the proposed work;
  it does not authorize implementation under Rule 2.
- Level ratios and eventual split weights remain provisional because APR P0
  has not resolved the Vacondio/PySPH stencil-convention mismatch.
- Dense per-level grids may fail the memory or timing gates and require a
  sparse key/sort alternative.

## Unresolved questions

- Whether dense per-level storage beats sparse storage on representative
  localized-refinement distributions.
- Whether the proposed single-level overhead and mixed-level timing gates are
  achievable with a generated loop over levels.
- Which production level ratio and split/merge policy will be selected after
  P0; this checkpoint deliberately does not decide them.

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > commit locally please.
- Timestamp: 2026-07-06T11:23:29+02:00
- Scope authorized: local commit of the plan/memory checkpoint only.
- Tier-2 implementation approval: still pending one of the exact Rule 2
  verdicts.
