#!/usr/bin/env python3
"""Performance + representative-snapshot comparison: Warp (fp32) vs PySPH (CPU).

Runs the *default* PySPH `dam_break_3d_lobovsky.py` Application (WCSPHScheme +
EPECIntegrator + WendlandQuintic, single-threaded Cython, fp64) and the additive
Warp `wc_sph_dam_break_step` (fp32, RTX GPU) for the SAME Lobovsky no-obstacle
case to the SAME physical time `tf`, then:

1. reports particle counts and wall-clock performance (total + per-step) and the
   Warp-vs-PySPH speedup, and
2. renders a side-by-side x-z snapshot of the final state (fluid coloured by
   speed, walls grey) for the two runs.

Timing notes (honest, reproducible -- not a tuned benchmark):
- CPU = the PySPH Application *solve loop* only (`app.setup()` then time
  `app.solve()`); compile/setup and a warm Cython cache are excluded.
- Warp = the GPU stepping loop only (NNPS build excluded; the first step carries
  a one-time module load, amortised in per-step over hundreds of steps).
- Both use their native adaptive dt + `n_damp`, so step counts differ slightly;
  per-step is the fixed-overhead-free metric. GPU advantage grows with particle
  count (this case is small; see the cross-GPU sweep for the 1M regime).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_dam_break_step
from pysph.solver.utils import get_files, load

from dam_break_3d_runner import REF_C0, WarpDamBreak3DRunner, damp_factor

EXAMPLE = 'pysph.examples.dam_break.dam_break_3d_lobovsky'


def run_cpu(dx, tf, pfreq, outdir):
    code = (
        "import time, json\n"
        "from {mod} import DamBreak3D\n"
        "app = DamBreak3D()\n"
        "app.setup(argv={argv!r})\n"
        "t0 = time.perf_counter()\n"
        "app.solve()\n"
        "dt = time.perf_counter() - t0\n"
        "print('PERFJSON ' + json.dumps({{'solve_s': dt, "
        "'count': int(app.solver.count), 't': float(app.solver.t)}}))\n"
    ).format(
        mod=EXAMPLE,
        argv=['--dx', str(dx), '-d', str(outdir), '--tf', str(tf),
              '--pfreq', str(pfreq), '--detailed-output'],
    )
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True)
    subprocess_wall = time.perf_counter() - t0
    perf = None
    for line in proc.stdout.splitlines():
        if line.startswith('PERFJSON '):
            perf = json.loads(line[len('PERFJSON '):])
    if perf is None:
        sys.stderr.write(proc.stdout[-2000:] + '\n' + proc.stderr[-2000:])
        raise SystemExit('CPU run did not report PERFJSON')
    perf['subprocess_wall_s'] = subprocess_wall
    return perf


def run_warp(dx, tf, args):
    import warp as wp
    runner = WarpDamBreak3DRunner(
        dx=dx, hdx=args.hdx, rho0=args.rho0, c0=args.c0, gamma=args.gamma,
        alpha=args.alpha, beta=args.beta, kernel='wendland',
        radius_scale=args.radius_scale, xsph_eps=args.xsph_eps, gz=args.gz,
        n_damp=args.n_damp, adaptive_dt=True, cfl=args.cfl,
    )
    fluid, wall = runner.create_particles()
    nnps = UniformGridWarpNNPS(
        dim=3, particles=[fluid, wall], radius_scale=runner.radius_scale
    )
    t, count, push = 0.0, 0, True
    wp.synchronize_device(nnps.device)
    t0 = time.perf_counter()
    while t < tf - 1.0e-12:
        scale = damp_factor(count, runner.n_damp)
        dt = wc_sph_dam_break_step(
            nnps, fluid_index=0, solid_indices=(1,), dt=runner.dt,
            rho0=runner.rho0, c0=runner.c0, gamma=runner.gamma,
            alpha=runner.alpha, beta=runner.beta, kernel='wendland',
            xsph_eps=runner.xsph_eps, gz=runner.gz, gravity_ramp=1.0,
            adaptive_dt=True, cfl=runner.cfl, dt_min=runner.dt_min,
            dt_max=runner.dt_max, adaptive_dt_scale=scale,
            step_dt_max=min(runner.dt_max, tf - t), push=push, return_dt=True,
        )
        push = False
        t += dt
        count += 1
    wp.synchronize_device(nnps.device)
    elapsed = time.perf_counter() - t0
    fluid.gpu.pull('x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p')
    wall.gpu.pull('x', 'y', 'z')
    return {
        'wall_s': elapsed, 'count': count, 't': t,
        'fluid': {k: np.asarray(getattr(fluid, k)) for k in
                  ('x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p')},
        'wall': {k: np.asarray(getattr(wall, k)) for k in ('x', 'y', 'z')},
        'n_fluid': fluid.get_number_of_particles(),
        'n_wall': wall.get_number_of_particles(),
    }


def _speed(d):
    return np.sqrt(d['u']**2 + d['v']**2 + d['w']**2)


def _panel(ax, fluid, wall, title, vmax, xlim, zlim):
    ax.scatter(wall['x'], wall['z'], s=4, c='0.78', marker='s',
               linewidths=0, label='wall')
    sc = ax.scatter(fluid['x'], fluid['z'], c=_speed(fluid), s=6, vmin=0.0,
                    vmax=vmax, cmap='viridis', linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    return sc


def plot_snapshot(cpu, warp, path, meta):
    cpu_speed = _speed(cpu['fluid'])
    warp_speed = _speed(warp['fluid'])
    vmax = max(float(cpu_speed.max()), float(warp_speed.max()), 1e-6)
    allx = np.concatenate([cpu['fluid']['x'], warp['fluid']['x'],
                           cpu['wall']['x'], warp['wall']['x']])
    allz = np.concatenate([cpu['fluid']['z'], warp['fluid']['z'],
                           cpu['wall']['z'], warp['wall']['z']])
    pad = 0.05
    xlim = (allx.min() - pad, allx.max() + pad)
    zlim = (allz.min() - pad, allz.max() + pad)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    _panel(axes[0], cpu['fluid'], cpu['wall'],
           'PySPH CPU (fp64)  t=%.3f s  %d fluid' % (
               cpu['t'], cpu['fluid']['x'].size), vmax, xlim, zlim)
    sc = _panel(axes[1], warp['fluid'], warp['wall'],
                'Warp GPU (fp32)  t=%.3f s  %d fluid' % (
                    warp['t'], warp['n_fluid']), vmax, xlim, zlim)
    fig.colorbar(sc, ax=axes, label='speed |v| (m/s)', shrink=0.85)
    fig.suptitle(meta, fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=0.08)
    p.add_argument('--tf', type=float, default=0.4)
    p.add_argument('--pfreq', type=int, default=1000000,
                   help='Large -> CPU dumps only initial + final (less IO).')
    p.add_argument('--hdx', type=float, default=1.3)
    p.add_argument('--rho0', type=float, default=1000.0)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--gamma', type=float, default=7.0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--xsph-eps', type=float, default=0.5)
    p.add_argument('--gz', type=float, default=-9.81)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--n-damp', type=int, default=50)
    p.add_argument('--radius-scale', type=float, default=2.0)
    p.add_argument('--prefix', default='cpu-vs-warp')
    p.add_argument('--keep-output', action='store_true')
    args = p.parse_args()
    args.dt_min = 0.0
    args.dt_max = None

    out_dir = Path(__file__).parent
    tmp = tempfile.mkdtemp(prefix='dam_break_perf_', dir=str(out_dir))

    cpu_perf = run_cpu(args.dx, args.tf, args.pfreq, tmp)
    snaps = get_files(str(tmp))
    if not snaps:
        raise SystemExit('No CPU snapshots produced')
    last = load(snaps[-1])
    arrays = last['arrays']
    fl, wl = arrays['fluid'], arrays.get('boundary', arrays.get('wall'))
    cpu = {
        't': float(last['solver_data']['t']),
        'fluid': {k: np.asarray(getattr(fl, k)) for k in
                  ('x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p')},
        'wall': {k: np.asarray(getattr(wl, k)) for k in ('x', 'y', 'z')},
    }

    warp = run_warp(args.dx, args.tf, args)

    img = out_dir / f'{args.prefix}-snapshot.png'
    meta = ('3D dam break (Lobovsky no-obstacle)  dx=%.3f  '
            'fluid=%d / wall=%d  c0=%.1f  Wendland' % (
                args.dx, warp['n_fluid'], warp['n_wall'], args.c0))
    plot_snapshot(cpu, warp, img, meta)

    cpu_steps = int(cpu_perf['count'])
    warp_steps = int(warp['count'])
    cpu_per_step = cpu_perf['solve_s'] / max(cpu_steps, 1)
    warp_per_step = warp['wall_s'] / max(warp_steps, 1)
    report = {
        'case': 'Lobovsky 3D dam-break, no obstacle',
        'reference_example': EXAMPLE,
        'dx': args.dx, 'tf': args.tf, 'c0': args.c0, 'kernel': 'wendland',
        'n_fluid': warp['n_fluid'], 'n_wall': warp['n_wall'],
        'n_total': warp['n_fluid'] + warp['n_wall'],
        'cpu_pysph_fp64': {
            'solve_s': cpu_perf['solve_s'], 'steps': cpu_steps,
            's_per_step': cpu_per_step, 'final_t': cpu['t'],
            'subprocess_wall_s': cpu_perf['subprocess_wall_s'],
        },
        'warp_gpu_fp32': {
            'wall_s': warp['wall_s'], 'steps': warp_steps,
            's_per_step': warp_per_step, 'final_t': warp['t'],
        },
        'speedup_wall': cpu_perf['solve_s'] / warp['wall_s'],
        'speedup_per_step': cpu_per_step / warp_per_step,
        'snapshot_image': str(img),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / f'{args.prefix}-perf.json').write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    if not args.keep_output:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
