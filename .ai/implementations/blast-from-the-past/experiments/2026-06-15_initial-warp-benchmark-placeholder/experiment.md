---
type: experiment
id: 2026-06-15_initial-warp-benchmark-placeholder
created: 2026-06-15T07:19:08 CET
author: @kunalpuri-prediqt
aspect: validation-benchmarks
status: active
last_checked: 2026-06-15T09:06:00 CET
---

# Experiment: Warp ParticleArray Mutation And Sync Baseline

## Purpose

Establish the first concrete correctness and timing baseline for the
NVIDIA-Warp-backed ParticleArray device mirror.

This experiment covers the ParticleArray operations that are most likely to
break when the backing storage moves from host arrays to device arrays:

- construction with `backend="warp"`
- full and selective host/device push/pull
- add particles
- remove particles
- remove tagged particles
- append particle arrays
- extract particles into a new array
- align local/remote/ghost particles
- preserve scalar and strided properties
- preserve constants and default values

## Setup

Run from the repository root on `prediqt-02`:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_initial-warp-benchmark-placeholder/run_correctness.sh
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_initial-warp-benchmark-placeholder/run_mutation_benchmark.sh
```

The scripts expect the PQT environment to be available through:

```bash
source "$HOME/prediqt/activate"
```

The current PQT environment has PySPH installed editable from this checkout,
Warp `1.14.0`, and standalone Zoltan installed under:

```text
/home/kunalp/prediqt/zoltan
```

## Hypothesis

Correctness should match the existing CPU ParticleArray semantics for all
covered mutation and synchronization operations.

Performance expectations are intentionally split:

- device-resident read/write operations should become the target for speedups;
- growth and deletion operations may not beat CPU yet, because the first
  implementation still uses host round-trips for some structural mutations;
- this baseline should expose the cost of those round-trips and guide the next
  Warp-kernel migration.

## Execution

### Correctness

`run_correctness.sh` runs:

- `pysph/base/tests/test_warp_device_helper.py`
- a focused CPU ParticleArray sanity slice for constructor, alignment,
  add-property, constants, remove, add, and extract behavior

Expected result:

```text
20 passed
7 passed
```

Warnings from Warp's Python 3.14 ctypes usage are acceptable for this baseline.

### Mutation Timing

`run_mutation_benchmark.sh` runs `benchmark_particle_mutations.py`, which times:

- add particles
- remove particles
- extract particles
- align particles
- full device-to-host pull after a device write

The benchmark records CPU and Warp timings for multiple particle counts. The
primary output is a readable table on stdout; redirect it when capturing a run:

```bash
bash .ai/implementations/blast-from-the-past/experiments/2026-06-15_initial-warp-benchmark-placeholder/run_mutation_benchmark.sh \
  | tee .ai/implementations/blast-from-the-past/experiments/2026-06-15_initial-warp-benchmark-placeholder/results-$(date +%Y%m%d-%H%M%S).txt
```

## Success Criteria

This experiment succeeds when:

- all focused Warp correctness tests pass;
- the CPU sanity slice still passes, proving the Warp path did not regress
  existing host semantics;
- every benchmark case returns correct particle counts and values;
- timings are captured with particle count, operation name, backend, repeat
  count, and p50 wall time;
- any operation that is slower on Warp is classified as either expected
  structural-mutation overhead or a follow-up optimization target.

This experiment does not yet require Warp mutation operations to be faster than
CPU. That threshold belongs to the next experiment after add/remove/extract are
moved away from host-side concatenation/readback and into device-side kernels.

## Results

Current known focused checks:

```text
python -m pytest -q pysph/base/tests/test_warp_device_helper.py
20 passed

CPU ParticleArray sanity slice
7 passed
```

Smoke timing is captured in `results-smoke-20260615.txt`. Larger timing runs
should be captured in `results-*.txt` files in this directory when performed.

## Conclusion

The first Warp ParticleArray implementation has correctness coverage for
particle add/delete-style operations. The missing piece was experiment
documentation and runnable measurement scripts, not test coverage.

## Follow-ups

- Run larger benchmark sizes on `prediqt-02`.
- Add a second experiment for device-side structural mutation kernels.
- Add a third experiment for NNPS-facing access patterns once the integration
  boundary moves from ParticleArray mirroring into neighbor search.
