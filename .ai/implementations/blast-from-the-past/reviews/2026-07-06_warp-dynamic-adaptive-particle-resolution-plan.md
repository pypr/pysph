---
type: review
date: 2026-07-06
user: @kunalpuri-prediqt
agent: codex
plan: 2026-07-06_warp-dynamic-adaptive-particle-resolution
adrs: []
aspects_touched: [warp-backend, gpu-nnps, particle-memory, validation-benchmarks, host-integration]
host_files: []
review_mode: prototype-owner
status: prototype-approved
---

# Review - Dynamic adaptive particle resolution plan package

## Diff summary

This prototype-governance and planning package adds:

- an approved Tier-2 plan for single-GPU, device-resident dynamic particle
  refinement/coarsening in the Warp WCSPH path;
- five separately reviewed checkpoints: baseline/design kill tests,
  multilevel NNPS, GPU particle lifecycle, conservative variable-resolution
  physics, and the adaptive fixed-obstacle dam-break;
- explicit correctness, conservation, performance, and scale gates;
- a session log recording consulted memory, scope, exact owner approval, and
  the next review gate;
- this pre-commit review artifact.
- a narrow owner-authorized prototype review mode in the implementation
  contract, with promotion/upstream work still gated by `@prabhu: LGTM`;
- matching review-template fields and validator checks for
  `status: prototype-approved`.

No PySPH runtime Python, Cython, Warp kernel, test, experiment output, or
host-project file is modified by this package. The memory validator is updated
only to enforce the new prototype-review metadata.

## Aspects touched and host files modified

| Aspect | Planning effect | Host files modified now |
|---|---|---|
| warp-backend | Defines additive APR sibling path and variable-resolution physics work | none |
| gpu-nnps | Defines exact multilevel/cross-level neighbor-search checkpoint | none |
| particle-memory | Defines device pool, scan allocation, and compaction checkpoint | none |
| validation-benchmarks | Defines obstacle baselines and scientific/scale gates | none |
| host-integration | Pins boundary and staged review/commit workflow | none |

The plan's future host-file list stays within the existing
`pysph/base/warp_*.py` and `pysph/base/tests/test_warp_*.py` boundary. The plan
requires an amendment before touching generic ParticleArray, Cython ABI,
solver, or shipped-example files.

## Behavioral / numerical changes

No solver or numerical behavior changes and no experiment starts. Governance
behavior changes: prototype commits meeting every Rule 4 restriction may use
quoted owner authorization instead of external `LGTM`; promotion and upstream
publication still require `@prabhu: LGTM`.

Future numerical thresholds are visible in the plan rather than implied:
mass/momentum conservation gates, uniform-fine observable comparisons, at
least 4x fewer active particles, and a target of at least 2x lower measured
time-to-solution. Failure to meet a performance target must be reported rather
than relabeled as success.

## Tests / validation run

No executable tests are required for this documentation-only package.

```text
$ git diff --check
<no output; exit 0>

$ python .ai/implementations/blast-from-the-past/scripts/validate-memory.py
validate-memory: PASS
```

## validate-memory.py

```text
validate-memory: PASS
```

## Boundary amendment

- implementation.md boundary section updated: n/a
- Amendments log entry: n/a

The proposed host files already match the approved Warp wildcard boundary.

## Visual aid

Waiver: this package contains no numerical or behavioral result; the phase and
acceptance-gate structure is clearer in the plan's ordered text than in a
duplicated diagram.

## Risks

- This approval covers a multi-session research track, but each checkpoint
  still needs its own review and exact `@prabhu: LGTM` before commit.
- The primary adaptive PySPH reference is 2D and EDAC-oriented; P0 explicitly
  prevents treating it as an already-validated 3D free-surface WCSPH method.
- The global finest-particle timestep may limit runtime gains even when memory
  and neighbor-work gains are substantial; the scale gates expose this risk.
- The plan is intentionally single-GPU. It does not claim that APR replaces
  later multi-GPU domain decomposition for arbitrarily large problems.
- A prototype exception could be abused to avoid review. The contract limits it
  to in-boundary Warp experimentation and makes the later promotion review
  cumulative and mandatory.

## Unresolved questions

- ADR-0007 must select the 3D split pattern, merge ownership, target-resolution
  representation, boundary policy, and canonical particle-pool layout after P0
  kill tests.
- Concrete uniform-equivalent scale depends on measured local GPU capacity and
  will be fixed in the P0 experiment rather than guessed in this review.

## Sign-off

- Review mode: prototype-owner
- Prototype owner: @kunalpuri-prediqt
- Prototype authorization, verbatim quote:
  > approved
- Timestamp: 2026-07-06T10:38:44 CEST

This authorization approves the proposed prototype exception, which explicitly
applies to the current APR plan package. It is not promotion approval.
