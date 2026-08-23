"""Benchmark PySPH CPU/Cython and Warp summation-density kernels."""

from __future__ import annotations

import argparse
import gc
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

from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import compute_summation_density
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.sph_evaluator import SPHEvaluator


@dataclass(frozen=True)
class Result:
    backend: str
    particles: int
    repeats: int
    p50_ms: float
    checksum: float
    status: str = "ok"


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
    m = np.ones(n) * dx * dx
    kwargs = dict(name='fluid', x=x, y=y, z=z, h=h, m=m)
    if backend is None:
        return [get_particle_array(**kwargs)]
    return [get_particle_array(backend=backend, **kwargs)]


def _time_cpu(particles: int, repeats: int) -> Result:
    arrays = _make_particles(particles)
    evaluator = SPHEvaluator(
        arrays=arrays,
        equations=[SummationDensity(dest='fluid', sources=['fluid'])],
        dim=2,
        kernel=CubicSpline(dim=2),
        backend='cython',
        nnps_factory=LinkedListNNPS,
    )
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        arrays[0].rho[:] = 0.0
        start = time.perf_counter()
        evaluator.evaluate(0.0, 0.1)
        samples.append((time.perf_counter() - start) * 1000.0)
        checksum = float(np.sum(arrays[0].rho))
    return Result(
        backend='cpu_cython',
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        checksum=checksum,
    )


def _time_warp(particles: int, repeats: int) -> Result:
    arrays = _make_particles(particles, backend='warp')
    nnps = UniformGridWarpNNPS(dim=2, particles=arrays, radius_scale=2.0)
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        rho = compute_summation_density(nnps, 0, 0)
        samples.append((time.perf_counter() - start) * 1000.0)
        checksum = float(wp.utils.array_sum(rho.dev))
    return Result(
        backend='warp_grid_density',
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        checksum=checksum,
    )


def _safe_time(fn, particles: int, repeats: int, backend: str) -> Result:
    try:
        return fn(particles, repeats)
    except Exception as exc:
        return Result(
            backend=backend,
            particles=particles,
            repeats=repeats,
            p50_ms=float('nan'),
            checksum=float('nan'),
            status=type(exc).__name__ + ": " + str(exc).splitlines()[0],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sizes', type=int, nargs='+',
        default=[1000000, 2000000, 5000000, 10000000],
    )
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument(
        '--backends', nargs='+',
        default=['cpu_cython', 'warp_grid_density'],
        choices=['cpu_cython', 'warp_grid_density'],
    )
    args = parser.parse_args()

    wp.init()
    print(f"# host: {platform.node()}", flush=True)
    print(f"# python: {sys.executable}", flush=True)
    print(f"# cpu: {_cpu_model()}", flush=True)
    print(f"# gpu: {_gpu_model()}", flush=True)
    print(f"# warp: {wp.__version__}", flush=True)
    print(
        "backend particles repeats p50_ms checksum speedup_vs_cpu status",
        flush=True,
    )
    for particles in args.sizes:
        results = {}
        if 'cpu_cython' in args.backends:
            results['cpu_cython'] = _safe_time(
                _time_cpu, particles, args.repeats, 'cpu_cython'
            )
        if 'warp_grid_density' in args.backends:
            results['warp_grid_density'] = _safe_time(
                _time_warp, particles, args.repeats, 'warp_grid_density'
            )

        cpu = results.get('cpu_cython')
        cpu_time = cpu.p50_ms if cpu is not None and cpu.status == 'ok' else None
        for backend in args.backends:
            result = results[backend]
            if cpu_time is None or result.status != 'ok':
                speedup = float('nan')
            else:
                speedup = cpu_time / result.p50_ms
            print(
                f"{result.backend:17s} "
                f"{result.particles:9d} "
                f"{result.repeats:7d} "
                f"{result.p50_ms:8.3f} "
                f"{result.checksum:14.6e} "
                f"{speedup:14.3f} "
                f"{result.status}",
                flush=True,
            )
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
