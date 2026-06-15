"""Benchmark device-side consumption of Warp NNPS neighbor caches."""

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

import warp as wp
from cyarray.carray import UIntArray

from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS


@dataclass(frozen=True)
class Result:
    backend: str
    particles: int
    repeats: int
    p50_ms: float
    avg_neighbor_sum: float
    checksum: float


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


def _make_particles(n: int, backend: str | None = None):
    rng = np.random.default_rng(12345)
    x = rng.random(n)
    y = rng.random(n)
    z = np.zeros(n)
    dx = np.sqrt(1.0 / n)
    h = np.ones(n) * 1.4 * dx
    m = np.ones(n)
    kwargs = dict(name='fluid', x=x, y=y, z=z, h=h, m=m)
    if backend is None:
        return [get_particle_array(**kwargs)]
    return [get_particle_array(backend=backend, **kwargs)]


def _cpu_neighbor_sum(arrays, n: int) -> tuple[float, float]:
    nnps = LinkedListNNPS(dim=2, particles=arrays, radius_scale=2.0)
    nbrs = UIntArray()
    total = 0.0
    mass = arrays[0].m
    nnps.set_context(0, 0)
    for d_idx in range(n):
        nnps.get_nearest_particles(0, 0, d_idx, nbrs)
        total += float(np.sum(mass[nbrs.get_npy_array()[:nbrs.length]]))
    return total / n, total


def _warp_neighbor_sum(arrays, n: int) -> tuple[float, float]:
    nnps = UniformGridWarpNNPS(dim=2, particles=arrays, radius_scale=2.0)
    out = nnps.compute_neighbor_sum(0, 0, 'm')
    checksum = float(wp.utils.array_sum(out))
    return checksum / n, checksum


def _time_backend(backend: str, particles: int, repeats: int) -> Result:
    samples = []
    avg_neighbor_sum = 0.0
    checksum = 0.0
    for _ in range(repeats):
        arrays = _make_particles(
            particles, backend='warp' if backend == 'warp_grid_reduce' else None
        )
        start = time.perf_counter()
        if backend == 'warp_grid_reduce':
            avg_neighbor_sum, checksum = _warp_neighbor_sum(arrays, particles)
        elif backend == 'cpu_reduce':
            avg_neighbor_sum, checksum = _cpu_neighbor_sum(arrays, particles)
        else:
            raise ValueError("Unknown backend: %s" % backend)
        samples.append((time.perf_counter() - start) * 1000.0)
    return Result(
        backend=backend,
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        avg_neighbor_sum=avg_neighbor_sum,
        checksum=checksum,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000000])
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    wp.init()

    print(f"# host: {platform.node()}")
    print(f"# python: {sys.executable}")
    print(f"# cpu: {_cpu_model()}")
    print(f"# gpu: {_gpu_model()}")
    print(f"# warp: {wp.__version__}")
    print(
        "backend particles repeats p50_ms avg_neighbor_sum checksum "
        "speedup_vs_cpu"
    )
    for particles in args.sizes:
        cpu_result = _time_backend('cpu_reduce', particles, args.repeats)
        warp_result = _time_backend(
            'warp_grid_reduce', particles, args.repeats
        )
        for result in (cpu_result, warp_result):
            speedup = cpu_result.p50_ms / result.p50_ms
            print(
                f"{result.backend:16s} "
                f"{result.particles:9d} "
                f"{result.repeats:7d} "
                f"{result.p50_ms:8.3f} "
                f"{result.avg_neighbor_sum:16.3f} "
                f"{result.checksum:12.3f} "
                f"{speedup:14.3f}"
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
