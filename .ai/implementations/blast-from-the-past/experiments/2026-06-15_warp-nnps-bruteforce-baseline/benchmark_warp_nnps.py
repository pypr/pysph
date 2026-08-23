"""Benchmark CPU linked-list NNPS against the first Warp brute-force NNPS."""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

from cyarray.carray import UIntArray

from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import BruteForceWarpNNPS, UniformGridWarpNNPS


@dataclass(frozen=True)
class Result:
    backend: str
    particles: int
    repeats: int
    p50_ms: float
    avg_neighbors: float


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _cpu_model() -> str:
    text = _run_text(["lscpu"])
    for line in text.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _gpu_model() -> str:
    text = _run_text([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    return text.splitlines()[0] if text else "unknown"


def _warp_version() -> str:
    try:
        import warp as wp
        return wp.__version__
    except Exception:
        return "unknown"


def _make_particles(n: int, backend: str | None = None):
    rng = np.random.default_rng(12345)
    x = rng.random(n)
    y = rng.random(n)
    z = np.zeros(n)
    dx = np.sqrt(1.0 / n)
    h = np.ones(n) * 1.4 * dx
    kwargs = dict(name='fluid', x=x, y=y, z=z, h=h)
    if backend is None:
        return [get_particle_array(**kwargs)]
    return [get_particle_array(backend=backend, **kwargs)]


def _query_all(nnps, n: int):
    nbrs = UIntArray()
    total = 0
    nnps.set_context(0, 0)
    for d_idx in range(n):
        nnps.get_nearest_particles(0, 0, d_idx, nbrs)
        total += nbrs.length
    return total / n


def _time_backend(backend: str, particles: int, repeats: int) -> Result:
    samples = []
    avg_neighbors = 0.0
    for _ in range(repeats):
        arrays = _make_particles(
            particles, backend='warp' if backend.startswith('warp') else None
        )
        if backend == 'warp':
            nnps = BruteForceWarpNNPS(dim=2, particles=arrays,
                                      radius_scale=2.0)
        elif backend == 'warp_cached':
            nnps = BruteForceWarpNNPS(dim=2, particles=arrays,
                                      radius_scale=2.0, cache=True)
        elif backend in ('warp_grid', 'warp_grid_device'):
            nnps = UniformGridWarpNNPS(dim=2, particles=arrays,
                                       radius_scale=2.0)
        else:
            nnps = LinkedListNNPS(dim=2, particles=arrays, radius_scale=2.0)

        start = time.perf_counter()
        if backend == 'warp_grid_device':
            cache = nnps.build_neighbor_cache_gpu(0, 0)
            avg_neighbors = cache['total_neighbors'] / particles
        else:
            avg_neighbors = _query_all(nnps, particles)
        samples.append((time.perf_counter() - start) * 1000.0)
    return Result(
        backend=backend,
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        avg_neighbors=avg_neighbors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument(
        '--backends', nargs='+',
        default=['cpu', 'warp', 'warp_cached', 'warp_grid'],
        choices=[
            'cpu', 'warp', 'warp_cached', 'warp_grid', 'warp_grid_device'
        ],
        help='Backends to run. For large sizes, use: cpu warp_grid_device.',
    )
    args = parser.parse_args()
    if 'cpu' not in args.backends:
        raise ValueError("cpu must be included to compute speedup_vs_cpu")

    print(f"# host: {platform.node()}")
    print(f"# python: {sys.executable}")
    print(f"# cpu: {_cpu_model()}")
    print(f"# gpu: {_gpu_model()}")
    print(f"# warp: {_warp_version()}")
    print("backend particles repeats p50_ms avg_neighbors speedup_vs_cpu")
    for particles in args.sizes:
        cpu_result = _time_backend('cpu', particles, args.repeats)
        results = [cpu_result]
        for backend in args.backends:
            if backend == 'cpu':
                continue
            results.append(_time_backend(backend, particles, args.repeats))
        for result in results:
            speedup = cpu_result.p50_ms / result.p50_ms
            print(
                f"{result.backend:12s} "
                f"{result.particles:9d} "
                f"{result.repeats:7d} "
                f"{result.p50_ms:8.3f} "
                f"{result.avg_neighbors:13.3f} "
                f"{speedup:14.3f}"
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
