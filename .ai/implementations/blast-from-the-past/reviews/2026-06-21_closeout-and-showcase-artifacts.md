---
type: review
date: 2026-06-21
user: @kunalpuri-prediqt
agent: codex
plan: n/a
adrs: [ADR-0005, ADR-0006]
aspects_touched: [validation-benchmarks, host-integration]
host_files: [CODEBASE_UNDERSTANDING.md]
status: approved
---

# Review - Remaining closeout and showcase artifacts

## Diff summary

This package versions every file intentionally left after the scoped P3 commit:

- marks eight completed foundational/elliptical experiments `complete` after
  checking their recorded outputs;
- records the developed `dx=0.025` 3D dam-break metrics in its experiment and
  review, with the JSON timing record and two actual-run images;
- versions the earlier 2026-06-20 daily/session closeouts;
- retains the coupled-run hero in its experiment directory as the source copy
  (the byte-identical review copy landed with P3);
- versions the owner-provided `CODEBASE_UNDERSTANDING.md` architecture report;
- updates current/closeout memory and the implementation boundary.

No Python/Cython/Warp source or test file changes in this package.

## Boundary amendment

`CODEBASE_UNDERSTANDING.md` is outside the previous host-file boundary. The
owner explicitly requested that all remaining files be committed, so
`implementation.md` now lists this root architecture snapshot as a retained
host-level reference. No executable behavior or public API is affected.

## Validation evidence

The experiment status audit only changes `status` and `last_checked` after
checking existing successful outputs. The dam-break additions record this
already-completed run:

```text
resolution / particles:  dx=0.025 / 125,687
Warp / CPU step:          0.03140 s / 0.42733 s
speedup:                  13.61x
steps / final time:       2,837 / 0.8016 s
all finite:               true
rho range:                989.7354 .. 1014.1267 kg/m^3
```

The coupled source image and committed P3 review image have identical SHA-256:

```text
f7cafa944a83b2bd459087357a707244462c4266be2d38588e24384418b055c9
```

```text
$ git diff --check
<no output; exit 0>

$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

No test rerun is required: this package contains no executable changes. The
immediately preceding P3 exact-tree run was `54 passed, 2 warnings`.

## Visual aid

Developed dam-break hero, actual Warp particles colored by speed:

![Warp GPU 3D dam-break collapse](2026-06-19_warp-3d-dam-break-lobovsky_assets/showcase-dx025-t040-hero.png)

Four-view developed-state verification:

![Warp GPU developed 3D dam-break](2026-06-19_warp-3d-dam-break-lobovsky_assets/showcase-dx025-t080-3d-snapshot.png)

## Risks / unresolved questions

- `CODEBASE_UNDERSTANDING.md` is a generated point-in-time report and may age;
  the curated `.ai/` spec remains the implementation's active memory.
- The coupled hero is intentionally stored twice (experiment source and review
  asset). The hashes prove identity; this costs about 1 MiB but preserves both
  provenance and review rendering.
- The 3D dam-break review is locally approved; upstream PR #435 review remains
  a separate publication workflow.

## Sign-off

- Reviewer: @prabhu
- Verdict, verbatim quote:
  > @prabhu: LGTM
- Timestamp: 2026-06-21T02:06:46 CEST
