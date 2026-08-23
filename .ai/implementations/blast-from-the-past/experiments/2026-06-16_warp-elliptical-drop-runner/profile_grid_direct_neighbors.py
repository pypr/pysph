#!/usr/bin/env python3
"""Segmented per-step profile of the grid-direct continuity path (ADR-0004).

Reuses the elliptical-drop initial condition from the Warp runner and steps the
continuity-density PEC integrator with fixed dt, instrumenting:

- whether any flat CSR neighbor cache is built on the continuity path
  (``build_neighbor_cache_gpu`` call count -- expected 0 under ADR-0004),
- the uniform-grid build cost (``_build_grid``, the remaining spatial index),
- the fused equation-kernel launch cost (``compute_wcsph_accel_continuity``),
- the steady-state per-step wall time.

This is the ADR-0004 counterpart to the ADR-0003 fused-equation segmented
profile: there the flat cache build dominated (~0.034-0.046 s/step); here it
should disappear from the continuity path entirely.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

import pysph.base.warp_sph as warp_sph
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_leapfrog_step
from warp_elliptical_drop_runner import WarpEllipticalDropRunner


def profile(nx, steps, warmup, dt):
    runner = WarpEllipticalDropRunner(
        nx=nx, steps=steps, dt=dt, rho0=1.0, c0=1400.0, p0=0.0, alpha=0.1,
        beta=0.0, eos='tait', gamma=7.0, kernel='gaussian', xsph_eps=0.5,
        density_mode='continuity',
    )
    pa = runner.create_particles()
    nnps = UniformGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=runner.radius_scale
    )

    counters = {'flat_cache_builds': 0, 'grid_builds': 0}
    timers = {'grid_build_s': [], 'equation_s': []}

    flat_orig = nnps.build_neighbor_cache_gpu
    grid_orig = nnps._build_grid
    eqn_orig = warp_sph.compute_wcsph_accel_continuity

    def counted_flat(src_index, dst_index):
        counters['flat_cache_builds'] += 1
        return flat_orig(src_index, dst_index)

    def timed_grid(src_index):
        # _build_grid syncs internally; time the (possibly cached) call.
        t0 = time.perf_counter()
        out = grid_orig(src_index)
        timers['grid_build_s'].append(time.perf_counter() - t0)
        counters['grid_builds'] += 1
        return out

    def timed_eqn(*args, **kwargs):
        t0 = time.perf_counter()
        out = eqn_orig(*args, **kwargs)
        timers['equation_s'].append(time.perf_counter() - t0)
        return out

    nnps.build_neighbor_cache_gpu = counted_flat
    nnps._build_grid = timed_grid
    warp_sph.compute_wcsph_accel_continuity = timed_eqn

    step_walls = []
    try:
        for s in range(steps):
            wp.synchronize_device(nnps.device)
            t0 = time.perf_counter()
            wc_sph_leapfrog_step(
                nnps, dt=dt, rho0=1.0, c0=1400.0, p0=0.0, alpha=0.1, beta=0.0,
                eos='tait', gamma=7.0, kernel='gaussian', xsph_eps=0.5,
                adaptive_dt=False, density_mode='continuity', return_dt=True,
            )
            wp.synchronize_device(nnps.device)
            step_walls.append(time.perf_counter() - t0)
    finally:
        warp_sph.compute_wcsph_accel_continuity = eqn_orig

    pa.gpu.pull('x', 'y', 'rho', 'u', 'v', 'au', 'av', 'arho')
    finite = all(
        np.all(np.isfinite(getattr(pa, n)))
        for n in ('x', 'y', 'rho', 'u', 'v', 'au', 'av', 'arho')
    )
    ke = 0.5 * float(np.sum(pa.m * (pa.u*pa.u + pa.v*pa.v)))

    steady = step_walls[warmup:]
    grid_steady = timers['grid_build_s'][warmup:]
    eqn_steady = timers['equation_s'][2 * warmup:]  # 2 launches/step

    def rng(xs):
        return [float(min(xs)), float(max(xs))] if xs else [0.0, 0.0]

    return {
        'particles': int(pa.get_number_of_particles()),
        'nx': nx,
        'steps': steps,
        'warmup_discarded': warmup,
        'dt': dt,
        'all_finite': bool(finite),
        'kinetic_energy': ke,
        'flat_cache_builds_total': counters['flat_cache_builds'],
        'grid_builds_total': counters['grid_builds'],
        'grid_builds_per_step': counters['grid_builds'] / steps,
        'equation_launches_total': len(timers['equation_s']),
        'step_wall_s_steady_range': rng(steady),
        'grid_build_s_steady_range': rng(grid_steady),
        'equation_launch_s_steady_range': rng(eqn_steady),
        'step_wall_s_all': [round(w, 6) for w in step_walls],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=565)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--dt', type=float, default=3.732778967800475e-07)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    result = profile(args.nx, args.steps, args.warmup, args.dt)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
