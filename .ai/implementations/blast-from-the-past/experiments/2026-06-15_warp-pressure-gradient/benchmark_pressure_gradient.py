"""Benchmark PySPH CPU/Cython and Warp pressure-gradient kernels."""

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
from pysph.base.warp_sph import compute_pressure_gradient
from pysph.sph.equation import Equation
from pysph.tools.sph_evaluator import SPHEvaluator


class PressureGradientOnly(Equation):
    """Pure inviscid pressure-gradient loop for CPU/Cython comparison."""

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, d_au, d_av, d_aw,
             s_m, s_rho, s_p, DWIJ):
        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])
        tmp = d_p[d_idx]*rhoi21 + s_p[s_idx]*rhoj21
        d_au[d_idx] += -s_m[s_idx] * tmp * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * tmp * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * tmp * DWIJ[2]


@dataclass(frozen=True)
class Result:
    backend: str
    particles: int
    repeats: int
    p50_ms: float
    au_checksum: float
    av_checksum: float
    aw_checksum: float
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
    rho = 1000.0 + 10.0 * rng.random(n)
    p = 100.0 * rng.random(n)
    zeros = np.zeros(n)
    kwargs = dict(
        name='fluid', x=x, y=y, z=z, h=h, m=m, rho=rho, p=p,
        au=zeros.copy(), av=zeros.copy(), aw=zeros.copy()
    )
    if backend is None:
        return [get_particle_array(**kwargs)]
    return [get_particle_array(backend=backend, **kwargs)]


def _time_cpu(particles: int, repeats: int) -> Result:
    arrays = _make_particles(particles)
    evaluator = SPHEvaluator(
        arrays=arrays,
        equations=[PressureGradientOnly(dest='fluid', sources=['fluid'])],
        dim=2,
        kernel=CubicSpline(dim=2),
        backend='cython',
        nnps_factory=LinkedListNNPS,
    )
    samples = []
    au_checksum = av_checksum = aw_checksum = 0.0
    for _ in range(repeats):
        arrays[0].au[:] = 0.0
        arrays[0].av[:] = 0.0
        arrays[0].aw[:] = 0.0
        start = time.perf_counter()
        evaluator.evaluate(0.0, 0.1)
        samples.append((time.perf_counter() - start) * 1000.0)
        au_checksum = float(np.sum(arrays[0].au))
        av_checksum = float(np.sum(arrays[0].av))
        aw_checksum = float(np.sum(arrays[0].aw))
    return Result(
        backend='cpu_cython',
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        au_checksum=au_checksum,
        av_checksum=av_checksum,
        aw_checksum=aw_checksum,
    )


def _time_warp(particles: int, repeats: int) -> Result:
    arrays = _make_particles(particles, backend='warp')
    nnps = UniformGridWarpNNPS(dim=2, particles=arrays, radius_scale=2.0)
    samples = []
    au_checksum = av_checksum = aw_checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        au, av, aw = compute_pressure_gradient(nnps, 0, 0)
        samples.append((time.perf_counter() - start) * 1000.0)
        au_checksum = float(wp.utils.array_sum(au.dev))
        av_checksum = float(wp.utils.array_sum(av.dev))
        aw_checksum = float(wp.utils.array_sum(aw.dev))
    return Result(
        backend='warp_grid_pgrad',
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
        au_checksum=au_checksum,
        av_checksum=av_checksum,
        aw_checksum=aw_checksum,
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
            au_checksum=float('nan'),
            av_checksum=float('nan'),
            aw_checksum=float('nan'),
            status=type(exc).__name__ + ": " + str(exc).splitlines()[0],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sizes', type=int, nargs='+',
        default=[1000000, 2000000, 5000000],
    )
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    wp.init()
    print(f"# host: {platform.node()}", flush=True)
    print(f"# python: {sys.executable}", flush=True)
    print(f"# cpu: {_cpu_model()}", flush=True)
    print(f"# gpu: {_gpu_model()}", flush=True)
    print(f"# warp: {wp.__version__}", flush=True)
    print(
        "backend particles repeats p50_ms au_checksum av_checksum "
        "aw_checksum speedup_vs_cpu status",
        flush=True,
    )
    for particles in args.sizes:
        cpu = _safe_time(_time_cpu, particles, args.repeats, 'cpu_cython')
        warp = _safe_time(_time_warp, particles, args.repeats,
                          'warp_grid_pgrad')
        for result in (cpu, warp):
            if cpu.status != 'ok' or result.status != 'ok':
                speedup = float('nan')
            else:
                speedup = cpu.p50_ms / result.p50_ms
            print(
                f"{result.backend:15s} "
                f"{result.particles:9d} "
                f"{result.repeats:7d} "
                f"{result.p50_ms:8.3f} "
                f"{result.au_checksum:14.6e} "
                f"{result.av_checksum:14.6e} "
                f"{result.aw_checksum:14.6e} "
                f"{speedup:14.3f} "
                f"{result.status}",
                flush=True,
            )
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
