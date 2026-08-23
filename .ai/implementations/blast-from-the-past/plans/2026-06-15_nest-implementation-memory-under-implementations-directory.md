---
type: plan
id: 2026-06-15_nest-implementation-memory-under-implementations-directory
author: @kunalpuri-prediqt
agent: codex
created: 2026-06-15T00:00:00 CET
status: approved
aspects: [host-integration]
host_files: []
within_boundary: true
---

# Plan: nest implementation memory under implementations directory

## Goal

Restructure the implementation memory so `blast-from-the-past` lives under `.ai/implementations/blast-from-the-past/`, with implementation-specific plans, decisions, ADR index/graph, reviews, updates, aspects, references, experiments, prompts, templates, skills, and scripts scoped inside that directory.

## Context

User preference: "id like that each implementation resides in its own directory under .ai so some thing like .ai/implementations/<some_new_work> and all the plans/decisions/adrs concerning that implementation are scoped within that directory".

Current scaffold places implementation memory directly under `.ai/`. This is fine for one implementation but will not scale cleanly if the repository has multiple independent implementation memories.

## Approach

1. Create `.ai/implementations/blast-from-the-past/`.
2. Move implementation-scoped directories/files into that directory:
   - `AGENTS.md`, `README.md`, `current.md`, `implementation.md`, `host-project-notes.md`, `conventions.md`, `glossary.md`
   - `plans/`, `decisions/`, `reviews/`, `updates/`, `skills/`, `aspects/`, `references/`, `experiments/`, `prompts/`, `templates/`, `scripts/`
3. Leave a minimal top-level `.ai/README.md` explaining the multi-implementation layout.
4. Add a top-level `.ai/AGENTS.md` router that points agents to `.ai/implementations/blast-from-the-past/AGENTS.md`.
5. Update the root `AGENTS.md` pointer to the nested implementation contract.
6. Update script path assumptions so scripts still find the implementation root when run from their nested location.
7. Update references inside moved Markdown files from `.ai/...` to `.ai/implementations/blast-from-the-past/...` where needed.
8. Update the pre-commit hook to run the nested validator.
9. Regenerate the nested decision index/graph.
10. Run:
    - `python .ai/implementations/blast-from-the-past/scripts/update-decision-graph.py`
    - `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`
    - `.git/hooks/pre-commit`
    - `git diff --check -- .ai AGENTS.md`

## Files expected to change

- `.ai/` memory files only.
- root `AGENTS.md` pointer.
- `.git/hooks/pre-commit`.
- No host application code.

## Tests / validation

- Nested decision graph regeneration must pass.
- Nested validator must pass.
- Pre-commit hook must pass.
- `git diff --check -- .ai AGENTS.md` must pass.

## Risks

- Script path assumptions may break after moving scripts deeper.
- Markdown links and contract text may still point at old top-level locations.
- The top-level `.ai` needs enough routing information to make future implementation selection obvious without duplicating implementation memory.

## Out of scope

- No changes to PySPH host application code.
- No changes to implementation aspects or technical scope.
- No commit.

## Estimated effort

M - broad file movement and script path updates, but no host code changes.

## Approval

- [x] Plan posted in chat
- Approved by: @kunalpuri-prediqt at 2026-06-15T07:31:16 CEST
- Approval, verbatim quote:
  > APPROVED
