---
type: decision
id: ADR-0001
date: 2026-06-15
author: @kunalpuri-prediqt
scope: global
status: Accepted
supersedes: []
relates_to: []
depends_on: []
conflicts_with: []
---

# ADR-0001: Adopt .ai memory system for blast-from-the-past

## Context

`blast-from-the-past` needs persistent implementation-scoped memory for NVIDIA Warp exploration inside PySPH without turning the memory into a catalogue of the entire host project.

## Decision

Adopt the `.ai/` memory system for `blast-from-the-past`, with ADR frontmatter as the decision source of truth, generated decision index/graph files, validation scripts, closeout discipline, experiment tracking, references, and a pre-commit hook.

## Rationale

The implementation has open-ended design and validation work across GPU backend choices, NNPS, particle memory, Cython boundaries, benchmarks, and host integration. A scoped memory system keeps those threads explicit and mechanically validated.

## Alternatives considered

- Use ad hoc chat history only: rejected because decisions and benchmark evidence would be hard to audit.
- Add permanent host docs immediately: rejected because the implementation is exploratory and needs working memory before stable documentation.

## Consequences

- Positive: plans, ADRs, closeouts, experiments, and reviews have a consistent place.
- Positive: validation checks catch malformed memory, stale decision graph files, boundary drift, and possible secrets.
- Negative: small process overhead before implementation work begins.

## Follow-ups

- Define concrete benchmark success criteria for "blazing fast particle dynamics."
- Capture exact NVIDIA Warp documentation/version and Prabhu guidance as references.
