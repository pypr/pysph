# Agent Operating Contract - blast-from-the-past

This implementation directory is the memory system for one implementation - **blast-from-the-past** - being developed inside the PySPH host project. It is not a memory system for the host. When this implementation ships, this implementation memory is archived into permanent docs, not deleted.

## Boot Sequence

1. Read `.ai/implementations/blast-from-the-past/current.md`.
2. If `current.md` lists in-flight experiments, inspect their actual outputs/logs and update each experiment's `status` and `last_checked` before doing anything else.
3. Read `.ai/implementations/blast-from-the-past/implementation.md`.
4. Read `.ai/implementations/blast-from-the-past/host-project-notes.md`.
5. Read the most recent daily closeout. On the first session of a new week, also read the most recent weekly closeout.
6. Identify which aspects today's task touches. Read those aspects' `context.md`, `open-questions.md`, `known-issues.md`, and only ADRs scoped to those aspects plus `global` via `.ai/implementations/blast-from-the-past/decisions/index.json`.
7. Open or create today's session log and maintain its `memory_consulted` list as `.ai/` files are opened.

State in chat that boot is complete before doing implementation work.

## Authority Order

1. Host project's pre-existing AI configs and conventions. No such configs were found during scaffold discovery; if added later, they override `.ai/`.
2. `.ai/implementations/blast-from-the-past/aspects/<aspect>/context.md` for aspect-scoped matters.
3. `.ai/implementations/blast-from-the-past/implementation.md` and `.ai/implementations/blast-from-the-past/conventions.md` for implementation-wide matters.
4. `.ai/implementations/blast-from-the-past/skills/*.md` for procedures.
5. Agent judgment.

If `.ai/` content contradicts a host config, the host wins; flag it and propose an ADR. If two `.ai/` files contradict each other, the most recently dated artifact wins; flag and repair stale memory in the same session.

## Five Non-Negotiable Rules

### Rule 1 - Closeouts Are Tagged by User

Every daily closeout, weekly closeout, and session log must begin with frontmatter containing a valid `user:` from:

- `@kunalpuri-prediqt`

The validator enforces this.

### Rule 2 - Plan Before Code, Sign-Off Required

Tier 0 trivial work: at most five lines, one file, no behavioral effect. No plan file; note it in the session log.

Tier 1 lightweight work: single session, roughly 50 LOC, at most three files, no ADR-worthy decision, within boundary. Post a one-paragraph plan in chat, wait for approval, and quote the user's approval verbatim with timestamp in the session log.

Tier 2 full work: write `.ai/implementations/blast-from-the-past/plans/{YYYY-MM-DD}_{slug}.md`, list aspects and host files, post it in chat, wait for `APPROVED`, `APPROVED WITH EDITS: ...`, or `REJECTED: ...`, and quote the approval verbatim in the plan.

Approval integrity: never paraphrase approval. An approval the agent cannot quote did not happen.

Experiments vs plans: parameter-only or config-only runs need an experiment entry. Persistent code changes need a plan.

Boundary visibility: if a Tier 2 plan touches host files outside `.ai/implementations/blast-from-the-past/implementation.md`, set `within_boundary: false` and call it out. Approved out-of-boundary work must amend the boundary during review.

### Rule 3 - Decisions Are Recorded as a Graph

For non-trivial design/modeling choices:

1. Create an ADR with `.ai/implementations/blast-from-the-past/scripts/new-decision.py`.
2. Run `python .ai/implementations/blast-from-the-past/scripts/update-decision-graph.py`.
3. Reference the ADR in plans, reviews, and session logs.

ADR frontmatter is the single source of truth. Do not hand-edit `.ai/implementations/blast-from-the-past/decisions/index.json` or `.ai/implementations/blast-from-the-past/decisions/graph.md`.

### Rule 4 - Review Before Commit

Before any commit:

1. Produce a review artifact in `.ai/implementations/blast-from-the-past/reviews/`.
2. Include diff summary, aspects, host files, behavioral/numerical changes, raw validation output, `validate-memory.py` output, risks, unresolved questions, and at least one visual aid or a one-line waiver.
3. Post the review in chat.
4. Wait for `@prabhu` to reply `LGTM` and quote the verdict verbatim in the review.
5. Only then commit. The commit message references the review and touched ADRs.

### Rule 5 - Do Not Cut Long-Running Tasks Short

For long tasks, proceed to completion or a defined checkpoint. Long numerical runs are handed off through experiment entries with `status: running`. If context is the genuine limit, write a session log with exact file:line, current test/experiment state, and the next concrete action.

## Closing Every Session

1. Finalize the session log with complete `memory_consulted`.
2. Update today's daily closeout.
3. Regenerate `.ai/implementations/blast-from-the-past/current.md` as a slim pointer.
4. Update aspect questions/issues and experiment statuses.
5. If it is Friday or the user says "wrap the week", produce the weekly closeout and run compaction.

## Curation

Weekly: run `python .ai/implementations/blast-from-the-past/scripts/compact.py`; daily closeouts and session logs older than 12 weeks move to `.ai/implementations/blast-from-the-past/updates/archive/`. Weeklies are never archived.

Monthly: refresh active aspect `## Current understanding` sections and bump `last_reviewed`.

Pruning signal: files absent from every `memory_consulted` for four or more weeks are pruning candidates.

## Secrets

Never copy credentials, tokens, API keys, connection strings, or private keys into `.ai/`. Reference locations by path only. `validate-memory.py` fails on detected secret patterns.
