"""Benchmark ParticleArray mutation and sync operations for CPU vs Warp.

This script is intentionally lightweight: it measures the API surface used by
the current prototype rather than trying to be a general benchmark harness.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

from pysph.base.utils import get_particle_array


@dataclass(frozen=True)
class Case:
    backend: str | None
    operation: str
    particles: int
    repeats: int
    p50_ms: float


def _make_array(n: int, backend: str | None):
    tag = np.zeros(n, dtype=np.int64)
    tag[n // 2::5] = 1
    tag[n // 3::7] = 2
    kwargs = dict(
        name=f"pa_{backend or 'cpu'}",
        x=np.linspace(0.0, 1.0, n),
        y=np.linspace(1.0, 2.0, n),
        z=np.linspace(2.0, 3.0, n),
        h=np.ones(n) * 0.1,
        m=np.ones(n),
        tag=tag,
    )
    if backend is None:
        return get_particle_array(**kwargs)
    return get_particle_array(backend=backend, **kwargs)


def _sync(backend: str | None):
    if backend == "warp":
        import warp as wp

        wp.synchronize()


def _time_case(
    backend: str | None,
    operation: str,
    particles: int,
    repeats: int,
    fn: Callable[[int, str | None], None],
) -> Case:
    samples = []
    for _ in range(repeats):
        _sync(backend)
        start = time.perf_counter()
        fn(particles, backend)
        _sync(backend)
        samples.append((time.perf_counter() - start) * 1000.0)
    return Case(
        backend=backend or "cpu",
        operation=operation,
        particles=particles,
        repeats=repeats,
        p50_ms=statistics.median(samples),
    )


def _case_add_particles(n: int, backend: str | None) -> None:
    pa = _make_array(n, backend)
    count = max(1, n // 10)
    pa.add_particles(
        x=np.linspace(4.0, 5.0, count),
        y=np.linspace(5.0, 6.0, count),
        z=np.linspace(6.0, 7.0, count),
        tag=np.zeros(count, dtype=np.int64),
        align=True,
    )
    expected = n + count
    actual = pa.get_number_of_particles()
    if actual != expected:
        raise AssertionError(f"add_particles expected {expected}, got {actual}")


def _case_remove_particles(n: int, backend: str | None) -> None:
    pa = _make_array(n, backend)
    remove = np.arange(1, n, 10, dtype=np.int64)
    pa.remove_particles(remove, align=True)
    expected = n - len(remove)
    actual = pa.get_number_of_particles()
    if actual != expected:
        raise AssertionError(f"remove_particles expected {expected}, got {actual}")


def _case_extract_particles(n: int, backend: str | None) -> None:
    pa = _make_array(n, backend)
    indices = np.arange(0, n, 11, dtype=np.int64)
    extracted = pa.extract_particles(indices, align=True)
    expected = len(indices)
    actual = extracted.get_number_of_particles()
    if actual != expected:
        raise AssertionError(f"extract_particles expected {expected}, got {actual}")


def _case_align_particles(n: int, backend: str | None) -> None:
    pa = _make_array(n, backend)
    pa.align_particles()
    pa.gpu.pull("tag") if backend == "warp" else None
    if pa.get_number_of_particles() != n:
        raise AssertionError("align_particles changed particle count")


def _case_pull_after_device_write(n: int, backend: str | None) -> None:
    pa = _make_array(n, backend)
    if backend != "warp":
        pa.x[:] = pa.x[:] + 2.0
        return
    pa.gpu.x.fill(3.5)
    pa.gpu.pull("x")
    if not np.allclose(pa.x[:], 3.5):
        raise AssertionError("device write did not pull back to host")


def _print_table(results: list[Case]) -> None:
    print("backend operation particles repeats p50_ms")
    for case in results:
        print(
            f"{case.backend:7s} "
            f"{case.operation:24s} "
            f"{case.particles:9d} "
            f"{case.repeats:7d} "
            f"{case.p50_ms:8.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    operations: list[tuple[str, Callable[[int, str | None], None]]] = [
        ("add_particles", _case_add_particles),
        ("remove_particles", _case_remove_particles),
        ("extract_particles", _case_extract_particles),
        ("align_particles", _case_align_particles),
        ("pull_after_device_write", _case_pull_after_device_write),
    ]

    results: list[Case] = []
    for particles in args.sizes:
        for name, fn in operations:
            for backend in (None, "warp"):
                results.append(
                    _time_case(
                        backend=backend,
                        operation=name,
                        particles=particles,
                        repeats=args.repeats,
                        fn=fn,
                    )
                )

    _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
