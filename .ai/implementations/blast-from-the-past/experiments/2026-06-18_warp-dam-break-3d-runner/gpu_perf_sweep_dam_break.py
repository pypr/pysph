#!/usr/bin/env python3
"""Cross-GPU performance sweep for the Warp 3D dam-break (fused) step.

Sweeps particle count (via ``dx`` over the Lobovsky no-obstacle geometry) and,
for each, measures the steady per-step wall time and throughput of the fused
``wc_sph_dam_break_step`` on the current GPU. Run it on each GPU and paste the
JSON block; the per-GPU JSONs together form a cross-GPU performance artifact
(mirrors ``gpu_perf_sweep.py`` for the elliptical drop).

GPU-only on purpose: the single-threaded PySPH CPU Application is slow at scale,
so the CPU-vs-Warp speedup is a separate 1M headline. Here the cross-GPU
comparison comes from running the same ``dx`` list on each GPU and comparing
throughput vs particle count.

Usage (run from the repo root, with the warp+pysph venv active):

    R=.ai/implementations/blast-from-the-past/experiments/2026-06-18_warp-dam-break-3d-runner
    python $R/gpu_perf_sweep_dam_break.py --label "L40S" --output sweep-dambreak-l40s.json

Notes:
- Physics matches the Lobovsky no-obstacle case: WendlandQuintic, Tait EOS,
  Tait-HG walls, gravity, fluid+wall arrays, radius_scale=2, fixed dt, fp32.
- The step is run with a FIXED dt (adaptive off) for clean per-step timing, like
  the elliptical sweep. The dt value does not affect per-step compute time.
- The initial condition is built with vectorized numpy masks (same box
  conditions as DamBreak3DGeometry) -- NOT the example's Python per-point loop --
  so building 1M-10M particle ICs is seconds, not minutes.
- The fused dam-break kernels cold-compile once on a cold disk cache (reported as
  cold_compile_s); a warmup run absorbs it so the swept points are warm.
- Out-of-memory at large dx is caught per point and recorded, so the sweep finds
  the GPU's capacity ceiling without aborting.
- dx -> particles (Lobovsky no-obstacle): 0.04 ~ 39k, 0.011 ~ 1.0M,
  0.009 ~ 1.8M, 0.006 ~ 5.6M, 0.005 ~ 9.5M (fluid + one wall array).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time

import numpy as np
import warp as wp

from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_dam_break_step

# Lobovsky no-obstacle geometry + physics (matches dam_break_3d_lobovsky.py).
H = 1.0
GRAVITY = 9.81
REF_C0 = 10.0 * math.sqrt(2.0 * GRAVITY * 0.55)
HDX = 1.3
NB_LAYERS = 1
RADIUS_SCALE = 2.0
DT = 1.0e-4
CONTAINER_H, CONTAINER_W, CONTAINER_L = 1.5 * H, H / 2.0, 161 * H / 30.0
FLUID_H, FLUID_W, FLUID_L = H, H / 2.0, 2.0 * H
PHYS = dict(rho0=1000.0, c0=REF_C0, gamma=7.0, alpha=0.25, beta=0.0,
            kernel='wendland', xsph_eps=0.5, gz=-GRAVITY)


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


def _fast_ic(dx):
    """Vectorized Lobovsky no-obstacle IC (same box masks as DamBreak3DGeometry)."""
    ghost = NB_LAYERS * dx
    cw2 = 0.5 * CONTAINER_W
    eps = 0.1 * dx
    xx, yy, zz = np.mgrid[-ghost:CONTAINER_L + ghost + eps:dx,
                          -cw2 - ghost:cw2 + ghost + eps:dx,
                          -ghost:CONTAINER_H + ghost + eps:dx]
    x, y, z = xx.ravel(), yy.ravel(), zz.ravel()
    fluid_m = ((x > 0) & (x <= FLUID_L) & (y > -cw2) & (y < cw2)
               & (z > 0) & (z <= FLUID_H))
    wall_m = ((y <= -cw2) | (y >= cw2) | (x >= CONTAINER_L) | (x <= 0)
              | (z <= 0))
    rho0 = PHYS['rho0']
    m0, h0 = rho0 * dx**3, HDX * dx

    def mk(mask, name):
        n = int(mask.sum())
        zr = np.zeros(n)
        return get_particle_array(
            name=name, x=x[mask].copy(), y=y[mask].copy(), z=z[mask].copy(),
            h=np.full(n, h0), m=np.full(n, m0), rho=np.full(n, rho0),
            p=zr.copy(), cs=np.full(n, PHYS['c0']),
            u=zr.copy(), v=zr.copy(), w=zr.copy(),
            au=zr.copy(), av=zr.copy(), aw=zr.copy(), arho=zr.copy(),
            ax=zr.copy(), ay=zr.copy(), az=zr.copy(),
            x0=zr.copy(), y0=zr.copy(), z0=zr.copy(),
            u0=zr.copy(), v0=zr.copy(), w0=zr.copy(), rho0=zr.copy(),
            backend='warp')

    return mk(fluid_m, 'fluid'), mk(wall_m, 'wall')


def _run_steps(nnps, steps):
    dev = nnps.device
    walls = []
    for i in range(steps):
        wp.synchronize_device(dev)
        t = time.perf_counter()
        wc_sph_dam_break_step(
            nnps, fluid_index=0, solid_indices=(1,), dt=DT, adaptive_dt=False,
            gravity_ramp=1.0, push=(i == 0), return_dt=True, **PHYS)
        wp.synchronize_device(dev)
        walls.append(time.perf_counter() - t)
    return walls


def bench_one(dx, steps, warmup):
    fluid, wall = _fast_ic(dx)
    nnps = UniformGridWarpNNPS(
        dim=3, particles=[fluid, wall], radius_scale=RADIUS_SCALE)
    nf = int(fluid.get_number_of_particles())
    nw = int(wall.get_number_of_particles())
    n = nf + nw

    walls = _run_steps(nnps, steps)
    fluid.gpu.pull('x', 'rho')
    finite = bool(np.all(np.isfinite(fluid.x)) and np.all(np.isfinite(fluid.rho)))

    steady = walls[warmup:] or walls
    median = statistics.median(steady)
    result = {
        'dx': dx, 'particles': n, 'fluid': nf, 'wall': nw,
        'per_step_s_min': round(min(steady), 6),
        'per_step_s_median': round(median, 6),
        'per_step_s_max': round(max(steady), 6),
        'throughput_particle_steps_per_s': round(n / median, 1),
        'all_finite': finite,
    }
    del nnps, fluid, wall
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dx-list',
        default='0.04,0.03,0.022,0.016,0.013,0.011,0.009,0.0075,0.006,0.005',
        help='comma-separated dx values (smaller dx = more particles)')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--label', default=None,
                        help='override the GPU label in the output')
    parser.add_argument('--output', default=None, help='also write JSON here')
    args = parser.parse_args()

    wp.init()
    gpu = _gpu_info()
    if args.label:
        gpu['name'] = args.label

    # Warm the kernel cache once (absorbs the cold compile of the fused kernels).
    t = time.perf_counter()
    try:
        bench_one(0.05, max(4, args.warmup + 2), args.warmup)
    except Exception as exc:  # pragma: no cover
        print('warmup failed:', exc, flush=True)
    cold_compile_s = round(time.perf_counter() - t, 2)
    print('cold_compile_s=%.2f' % cold_compile_s, flush=True)

    dx_list = [float(s) for s in args.dx_list.split(',') if s.strip()]
    results = []
    for dx in dx_list:
        try:
            r = bench_one(dx, args.steps, args.warmup)
            results.append(r)
            print('dx=%-6.4f particles=%-9d per_step=%.6f s  '
                  'throughput=%.3e p-steps/s  finite=%s'
                  % (r['dx'], r['particles'], r['per_step_s_median'],
                     r['throughput_particle_steps_per_s'], r['all_finite']),
                  flush=True)
        except Exception as exc:
            results.append({'dx': dx, 'error': type(exc).__name__ + ': '
                            + str(exc)[:120]})
            print('dx=%-6.4f FAILED: %s' % (dx, type(exc).__name__), flush=True)
            gc.collect()

    report = {
        'benchmark': 'warp 3D dam-break fused WCSPH step (fp32)',
        'hardware': gpu,
        'config': {
            'dt': DT, 'steps': args.steps, 'warmup_discarded': args.warmup,
            'physics': PHYS, 'radius_scale': RADIUS_SCALE, 'hdx': HDX,
            'geometry': 'Lobovsky no-obstacle', 'cold_compile_s': cold_compile_s,
        },
        'results': results,
    }

    print('\n=== sweep table ===')
    print('| dx | particles | per-step (s) | throughput (p-steps/s) | finite |')
    print('|---:|---:|---:|---:|:--:|')
    for r in results:
        if 'error' in r:
            print('| %.4f | -- | ERROR | %s | -- |' % (r['dx'], r['error']))
        else:
            print('| %.4f | %d | %.6f | %.3e | %s |' % (
                r['dx'], r['particles'], r['per_step_s_median'],
                r['throughput_particle_steps_per_s'], r['all_finite']))

    print('\n=== JSON (paste this) ===')
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
