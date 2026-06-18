# Building PySPH (with the Warp GPU backend)

This document describes how to build and run this repository the way it was set
up on the development machine used for the Warp-backend work, so you can
reproduce it elsewhere.

PySPH is a Python package with **Cython/C++ extension modules** that must be
compiled. The experimental **Warp GPU backend** (`pysph/base/warp_*.py`) is an
*extra* on top of upstream PySPH: it is not in the repo's requirements and needs
`warp-lang` plus an NVIDIA GPU.

There are two tiers you can build:

- **Core PySPH** — the CPU package (Cython extensions + the standard solver).
  Enough to run `pysph/examples/...` and the non-GPU tests.
- **Warp GPU backend** — adds `warp-lang` and a CUDA-capable GPU, needed to run
  `pysph/base/warp_*.py`, `pysph/base/tests/test_warp_*.py`, and the
  elliptical-drop GPU benchmarks under
  `.ai/implementations/blast-from-the-past/experiments/`.

---

## 1. Prerequisites

| Component | What's needed | Version used here (known-good) |
|---|---|---|
| OS | Linux (incl. WSL2); macOS/Windows also supported by PySPH upstream | Linux 6.6 (WSL2) |
| C/C++ compiler | A working `gcc`/`g++` (OpenMP optional but recommended) | gcc/g++ 15.2.0 |
| Python | CPython 3.x with `venv` | 3.14.4 |
| Core Python build deps | numpy, Cython, compyle, cyarray, mako, pytools, Beaker, setuptools, wheel | see table in §6 |
| **GPU extras** | NVIDIA GPU + driver, `warp-lang` | RTX 4060 Laptop (sm_89), driver 595.79, CUDA 13.2 |

Notes:

- The base scientific stack on the dev machine (numpy, h5py, mpi4py, pybind11)
  was provided by **Spack** and a pip venv was layered on top. That is **not
  required** — plain `pip` works for the core build. `h5py`/`mpi4py` are only
  needed for HDF5 output and MPI/Zoltan parallel runs respectively; skip them
  for a minimal build.
- Warp ships its own CUDA runtime support; you do **not** need a separately
  installed CUDA Toolkit, only a recent NVIDIA driver and a supported GPU.

---

## 2. Get the source

```bash
git clone <this-repo-url> pysph
cd pysph
```

---

## 3. Create an isolated environment

Any Python 3.10+ works; 3.14 was used here. A plain venv is sufficient.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

(Conda/mamba work too; this repo does not require them. The dev machine used a
plain venv at `~/.pqt_venv_<hash>/`, no conda.)

---

## 4. Build core PySPH (Cython extensions)

Install the build/runtime dependencies, then compile the extension modules
in place. The canonical build command (also `make build`) is:

```bash
pip install -r requirements.txt          # numpy, Cython, compyle, cyarray, mako, pytools, Beaker, setuptools
python setup.py build_ext --inplace      # compile the Cython/C++ extensions
```

Or do both in one step with an editable install (uses `pyproject.toml`'s
build backend and builds the extensions):

```bash
pip install -e .
```

This produces ~22 compiled modules under `pysph/` (e.g.
`pysph/base/nnps_base.*.so`, `linked_list_nnps`, `octree`, `c_kernels`,
`gpu_nnps_base`, ...). If `build_ext` reports it cannot import Cython it
disables OpenMP and continues; install `Cython>=0.20` (3.2.5 here) to keep
OpenMP and the threaded NNPS paths.

Optional CPU extras:

```bash
pip install -r requirements-test.txt     # pytest, mock, h5py, vtk (tests + HDF5 + viz)
```

---

## 5. Add the Warp GPU backend (extra)

Only needed to run the Warp code and GPU benchmarks. Requires an NVIDIA GPU and
a recent driver.

```bash
pip install warp-lang        # 1.14.0 used here
```

The Warp device path runs in **single precision (fp32)** because compyle's
configuration has `use_double = False` on this machine. Verify with:

```bash
python -c "from compyle.config import get_config; print('use_double =', get_config().use_double)"
# -> use_double = False
```

Double precision is governed by compyle's configuration; see the compyle docs if
you need fp64 on the GPU path.

For specific data-center and Blackwell GPUs (V100, A100, H100, RTX 5090,
RTX PRO 6000 Blackwell), see [§10](#10-gpu-architecture-compatibility-v100-a100-h100-rtx-5090-rtx-pro-6000-blackwell).

---

## 6. Verify the build

Core imports and the Warp backend:

```bash
python -c "import pysph, compyle; print('pysph + compyle import OK')"
python -c "import warp; warp.init()"     # GPU path: should list your CUDA device
```

Run the test suites:

```bash
# Standard PySPH CPU tests (excludes slow tests by default; see setup.cfg/tox.ini)
python -m pytest -m "not slow" pysph

# Warp GPU backend focused suite (requires GPU + warp-lang) -- 50 tests here
python -m pytest -q \
    pysph/base/tests/test_warp_codegen.py \
    pysph/base/tests/test_warp_sph.py \
    pysph/base/tests/test_warp_nnps.py
```

The first Warp run JIT-compiles kernels (slow, a few minutes cold); compiled
kernels are cached under `~/.cache/warp/<version>/`, so reruns are fast.

Known-good toolchain on the dev machine (from `pip freeze`):

| Package | Version |
|---|---|
| Python | 3.14.4 |
| numpy | 2.4.6 |
| Cython | 3.2.5 |
| compyle | 0.9.1 |
| cyarray | 1.2 |
| mako | 1.3.12 |
| pytools | (per requirements) |
| warp-lang | 1.14.0 (GPU extra) |
| matplotlib | 3.10.9 (benchmark plots) |
| h5py | 3.16.0 (HDF5 output, optional) |
| mpi4py | 4.1.1 (parallel, optional) |

---

## 7. Running the Warp elliptical-drop benchmarks (optional)

These live under the implementation tree and import the runner by path, so set
`PYTHONPATH` to the experiment directory:

```bash
R=.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner

# Warp-only fixed-step run
PYTHONPATH=$R python $R/warp_elliptical_drop_runner.py \
    --nx 565 --steps 100 --dt 3.732778967800475e-07 \
    --c0 1400.0 --alpha 0.1 --eos tait --gamma 7.0 \
    --kernel gaussian --xsph-eps 0.5 --density-mode continuity

# CPU (PySPH Application) vs Warp headline, million particles, 100 fixed steps
PYTHONPATH=$R python $R/headline_million_100step.py --nx 565 --steps 100

# Segmented per-step profile of the grid-direct neighbor path (ADR-0004)
PYTHONPATH=$R python $R/profile_grid_direct_neighbors.py --nx 565 --steps 12 --warmup 2
```

The CPU baseline in the headline/comparison scripts is the real PySPH Cython
Application (`pysph/examples/elliptical_drop_no_scheme.py`), run single-threaded
with `--no-adaptive-timestep --n-damp 0` for an apples-to-apples fixed-step
comparison.

---

## 8. Rebuilding after editing Cython (`.pyx`/`.pxd`)

Re-run the in-place build; only changed modules recompile:

```bash
python setup.py build_ext --inplace      # or: make build
```

Editing the pure-Python Warp files (`pysph/base/warp_*.py`) needs no rebuild.

---

## 9. Gotchas

- **`python` vs `python3`**: this repo's tooling and the Makefile call bare
  `python`. Inside an activated venv `python` resolves correctly. In a
  non-interactive shell where only `python3` is on `PATH`, invoke the venv
  interpreter explicitly (e.g. `</path/to/.venv>/bin/python`).
- **Pre-commit hook**: `.git/hooks/pre-commit` runs
  `python .ai/implementations/blast-from-the-past/scripts/validate-memory.py`.
  If bare `python` is not on `PATH` the hook fails; run the validator manually
  with the venv interpreter and commit with `git commit --no-verify`.
- **OpenMP**: if Cython is missing at build time, `setup.py` disables OpenMP and
  falls back to `no_omp_threads`. Install Cython before building for threaded
  NNPS.
- **GPU not found**: `warp.init()` prints the detected devices. If it shows only
  `cpu`, check the NVIDIA driver (`nvidia-smi`) and that `warp-lang` matches your
  driver's capability.

---

## 10. GPU architecture compatibility (V100, A100, H100, RTX 5090, RTX PRO 6000 Blackwell)

**The PySPH build is GPU-architecture-independent.** The compiled Cython
extensions are CPU code, and the Warp backend JIT-compiles its kernels to your
GPU's architecture at runtime (cached under `~/.cache/warp/<version>/`, keyed by
architecture). So **no build or source changes are needed for any of these
GPUs** — moving to a different card only requires:

1. an NVIDIA driver new enough for that architecture, and
2. a `warp-lang` whose bundled CUDA toolkit supports it.

The `warp-lang 1.14.0` used here bundles **CUDA Toolkit 12.9**, which targets
every architecture below (sm_70 through sm_120). Confirmed live via
`warp.init()`: `CUDA Toolkit 12.9, Driver 13.2`. So the same
`pip install warp-lang` step from §5 works on all of them.

| GPU | Architecture | Compute capability | VRAM | Min NVIDIA driver branch | fp64 throughput |
|---|---|---|---|---|---|
| Tesla V100 | Volta | `sm_70` | 16 / 32 GiB | R384+ (any current) | strong (~1:2) |
| A100 | Ampere | `sm_80` | 40 / 80 GiB | R450+ | strong (~1:2) |
| H100 | Hopper | `sm_90` | 80 / 94 GiB | R525+ | strong (~1:2) |
| RTX 5090 | Blackwell (consumer) | `sm_120` | 32 GiB | **R570+** | weak (~1:64) |
| RTX PRO 6000 Blackwell | Blackwell (workstation) | `sm_120` | 96 GiB | **R570+** | weak |
| RTX 4060 (dev machine) | Ada Lovelace | `sm_89` | 8 GiB | — | weak |

Per-architecture notes:

- **Drivers.** Blackwell (`sm_120`: RTX 5090, RTX PRO 6000 Blackwell) requires an
  **R570 or newer** driver; older drivers will not enumerate the GPU. V100 /
  A100 / H100 work with any reasonably current driver. The dev machine's driver
  reports CUDA 13.2 (newer than R570), so it already covers Blackwell.
- **warp-lang version.** 1.14.0 (CUDA Toolkit 12.9) covers all of these.
  **CUDA 12.8 was the first toolkit with Blackwell (`sm_100`/`sm_120`) support**,
  so if you are on an older `warp-lang` whose bundled CUDA predates 12.8, the
  RTX 5090 / RTX PRO 6000 will fail to compile kernels — run
  `pip install -U warp-lang` to get a CUDA ≥ 12.8 build. V100/A100/H100 are fine
  on much older warp-lang.
- **First run recompiles, no code change.** Because the kernel cache is keyed by
  architecture, the first run on a new GPU JIT-compiles (a few minutes) and then
  caches. Nothing in `pysph/base/warp_*.py` changes per GPU.
- **Precision.** This code runs **fp32** (compyle `use_double = False`). On
  V100 / A100 / H100 (full-rate fp64, ~1:2) you can enable double precision via
  compyle's config with little throughput penalty if you need it. On the
  RTX 5090 and RTX PRO 6000 Blackwell, fp64 is heavily rate-limited (~1:64) —
  **keep fp32** there.
- **Memory / problem size.** The elliptical-drop benchmarks use a small fraction
  of the 4060's 8 GiB at 1M particles. With 32–96 GiB (Blackwell) or 80 GiB
  (A100) / 80–94 GiB (H100) you can scale particle counts up by one to two orders
  of magnitude. Note the initial particle layout is built with `numpy.mgrid` on
  the **CPU** before upload, so size host RAM accordingly.
- **Multi-GPU.** The current Warp NNPS/SPH path targets a **single device**
  (`cuda:0` by default). On a multi-GPU node (e.g. 8×A100 / 8×H100) it uses one
  GPU; pin a specific one with `CUDA_VISIBLE_DEVICES=<n>` or by constructing the
  NNPS with the desired `wp` device. Multi-GPU domain decomposition is not
  implemented in this backend.

Verify on the target machine before running:

```bash
python -c "import warp as wp; wp.init(); print([(d.name, 'sm_%s' % d.arch, round(d.total_memory/2**30, 1)) for d in wp.get_cuda_devices()])"
```

If each device shows the expected `sm_XX` it will run. An "unsupported
architecture" / PTX / NVRTC error means the `warp-lang` or driver is too old for
that GPU — update both (newest driver for the card, `pip install -U warp-lang`).
