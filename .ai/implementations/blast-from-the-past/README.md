# .ai/ - Memory for blast-from-the-past

Tracks the development of one implementation - **blast-from-the-past** - added to this repository. Additive: nothing outside `.ai/` changed at install except this repository's permitted root pointer and the permitted pre-commit hook.

## For Humans

- `implementation.md` - central spec, including the integration boundary and amendments log.
- `AGENTS.md` - operating contract every agent follows.
- `current.md` - slim live status, regenerated every closeout.
- `aspects/<name>/` - context, open questions, issues, and notes for each dimension of the work.
- `decisions/graph.md` - generated from ADR frontmatter; never hand-edit it or `index.json`.
- `experiments/` - tracked runs, baselines, and validation studies.
- `references/` - annotated literature, human guidance, and API references.

## Validation

Run:

```bash
python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
```

The same command runs from the installed pre-commit hook.

## When This Implementation Is Done

Archive `.ai/implementations/blast-from-the-past/` into the host's permanent docs or under the implementation's directory. Do not delete it: the decision graph and validation history are part of the artifact.
