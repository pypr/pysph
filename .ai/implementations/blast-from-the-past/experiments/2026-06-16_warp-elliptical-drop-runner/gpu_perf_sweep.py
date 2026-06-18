#!/usr/bin/env python3
"""GPU performance sweep for the Warp grid-direct WCSPH backend.

Sweeps particle count (via ``nx``) and, for each, measures the steady per-step
wall time and throughput of the fixed-step continuity-density PEC step on the
current GPU. Run it on each GPU and paste the JSON block; the per-GPU JSONs
together form a cross-GPU performance artifact.

This is GPU-only on purpose -- the single-threaded PySPH CPU Application is slow
at scale, so the CPU-vs-Warp speedup lives in ``headline_million_100step.py``
(one point at 1M). Here the cross-GPU comparison comes from running the same
sweep on each GPU and comparing throughput vs particle count.

Usage (run from the repo root):

    R=.ai/implementations/blast-from-the-past/experiments/2026-06-16_warp-elliptical-drop-runner
    PYTHONPATH=$R python $R/gpu_perf_sweep.py
    # custom sweep / longer averaging / a label:
    PYTHONPATH=$R python $R/gpu_perf_sweep.py \
        --nx-list 100,200,400,565,1000,1400,1800,2600 --steps 16 --warmup 6 \
        --label "RTX PRO 6000 Blackwell" --output sweep.json

Notes:
- Physics matches the production runs: gaussian kernel, Tait EOS, continuity
  density, radius_scale=3, fixed dt, fp32 (compyle use_double=False).
- The fused grid kernel cold-compiles once on a cold disk cache (reported as
  cold_compile_s); a warmup run absorbs it so the swept points are warm.
- Out-of-memory at large nx is caught per point and recorded, so the sweep
  finds the GPU's capacity ceiling without aborting.
- nx maps to particle count via the elliptical-drop disk fill
  (particles ~ pi * nx^2): nx=100 ~ 31k, 565 ~ 1.0M, 1000 ~ 3.1M, 1800 ~ 10M.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import numpy as np
import warp as wp

from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_leapfrog_step
from warp_elliptical_drop_runner import WarpEllipticalDropRunner

DT = 3.732778967800475e-07
PHYS = dict(
    rho0=1.0, c0=1400.0, p0=0.0, alpha=0.1, beta=0.0, eos='tait', gamma=7.0,
    kernel='gaussian', xsph_eps=0.5, density_mode='continuity',
)


def _gpu_info():
    try:
        devs = wp.get_cuda_devices()
    except Exception:
        devs = []
    if not devs:
        return {'name': 'cpu/none', 'arch': 'n/a', 'memory_gib': None}
    d = devs[0]
    return {
        'name': d.name,
        'arch': 'sm_%s' % d.arch,
        'memory_gib': round(d.total_memory / 2**30, 1),
        'warp': wp.__version__,
    }


def _run_steps(nnps, steps):
    dev = nnps.device
    walls = []
    for _ in range(steps):
        wp.synchronize_device(dev)
        t = time.perf_counter()
        wc_sph_leapfrog_step(nnps, dt=DT, return_dt=True, **PHYS)
        wp.synchronize_device(dev)
        walls.append(time.perf_counter() - t)
    return walls


def bench_one(nx, steps, warmup):
    runner = WarpEllipticalDropRunner(nx=nx, steps=steps, dt=DT, **PHYS)
    pa = runner.create_particles()
    nnps = UniformGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=runner.radius_scale
    )
    n = int(pa.get_number_of_particles())
    t_setup = time.perf_counter()
    wp.synchronize_device(nnps.device)
    setup_s = time.perf_counter() - t_setup

    walls = _run_steps(nnps, steps)
    pa.gpu.pull('x', 'y', 'rho', 'u', 'v')
    finite = bool(np.all(np.isfinite(pa.x)) and np.all(np.isfinite(pa.rho)))

    steady = walls[warmup:] or walls
    median = statistics.median(steady)
    result = {
        'nx': nx,
        'particles': n,
        'setup_s': round(setup_s, 4),
        'per_step_s_min': round(min(steady), 6),
        'per_step_s_median': round(median, 6),
        'per_step_s_max': round(max(steady), 6),
        'throughput_particle_steps_per_s': round(n / median, 1),
        'all_finite': finite,
    }
    del nnps, pa, runner
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nx-list', default='100,200,300,400,565,800,1000,1400,1800',
        help='comma-separated nx values (particles ~ pi*nx^2)')
    parser.add_argument('--steps', type=int, default=14)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--label', default=None,
                        help='override the GPU label in the output')
    parser.add_argument('--output', default=None, help='also write JSON here')
    args = parser.parse_args()

    wp.init()
    gpu = _gpu_info()
    if args.label:
        gpu['name'] = args.label

    # Warm the kernel cache once (absorbs the one-time cold compile) and time it.
    t = time.perf_counter()
    try:
        bench_one(20, max(4, args.warmup + 2), args.warmup)
    except Exception as exc:  # pragma: no cover
        print('warmup failed:', exc)
    cold_compile_s = round(time.perf_counter() - t, 2)

    nx_list = [int(s) for s in args.nx_list.split(',') if s.strip()]
    results = []
    for nx in nx_list:
        try:
            r = bench_one(nx, args.steps, args.warmup)
            results.append(r)
            print('nx=%-5d particles=%-9d per_step=%.6f s  throughput=%.3e p-steps/s  finite=%s'
                  % (r['nx'], r['particles'], r['per_step_s_median'],
                     r['throughput_particle_steps_per_s'], r['all_finite']),
                  flush=True)
        except Exception as exc:
            results.append({'nx': nx, 'error': type(exc).__name__ + ': '
                            + str(exc)[:120]})
            print('nx=%-5d FAILED: %s' % (nx, type(exc).__name__), flush=True)
            gc.collect()

    report = {
        'benchmark': 'warp grid-direct WCSPH continuity PEC step (fp32)',
        'hardware': gpu,
        'config': {
            'dt': DT, 'steps': args.steps, 'warmup_discarded': args.warmup,
            'physics': PHYS, 'cold_compile_s': cold_compile_s,
        },
        'results': results,
    }

    print('\n=== sweep table ===')
    print('| nx | particles | per-step (s) | throughput (particle-steps/s) | finite |')
    print('|---:|---:|---:|---:|:--:|')
    for r in results:
        if 'error' in r:
            print('| %d | -- | ERROR | %s | -- |' % (r['nx'], r['error']))
        else:
            print('| %d | %d | %.6f | %.3e | %s |' % (
                r['nx'], r['particles'], r['per_step_s_median'],
                r['throughput_particle_steps_per_s'], r['all_finite']))

    print('\n=== JSON (paste this) ===')
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
