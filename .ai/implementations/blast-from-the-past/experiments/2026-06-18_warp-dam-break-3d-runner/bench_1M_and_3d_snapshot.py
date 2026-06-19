#!/usr/bin/env python3
"""~1M-particle perf comparison + a 3D-explicit developed snapshot.

Two deliverables for the 3D dam-break (Lobovsky no-obstacle):

1. **Perf at ~1M particles** -- per-step throughput of the default PySPH CPU
   Application (fp64, single-threaded Cython) vs the Warp `wc_sph_dam_break_step`
   (fp32 GPU), each over a fixed short step count. Per-step / particle-steps-per-
   second is the meaningful large-N metric (a *developed* 1M run to a physical
   `tf` is multi-hour on the CPU, so it is intentionally not attempted -- matching
   how the committed 1M elliptical comparison was measured).

2. **A 3D-explicit snapshot** of the developed Warp 1M state (x-z side view, x-y
   top-down, and a 3D scatter), so the fully-3D nature of the simulation is
   unmistakable (the simulation has many particle layers across the channel
   width `y`; the earlier single x-z panel only projected them).

The perf JSON is printed and written BEFORE the (longer) developed-snapshot phase
so the headline number is captured regardless of the render phase.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np

from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_dam_break_step
from dam_break_3d_runner import REF_C0, WarpDamBreak3DRunner, damp_factor

EXAMPLE = 'pysph.examples.dam_break.dam_break_3d_lobovsky'


def cpu_per_step(dx, steps, outdir):
    """Time the PySPH Application solve loop over a fixed step count at ~1M."""
    code = (
        "import time, json\n"
        "from {mod} import DamBreak3D\n"
        "app = DamBreak3D()\n"
        "app.setup(argv={argv!r})\n"
        "t0 = time.perf_counter()\n"
        "app.solve()\n"
        "dt = time.perf_counter() - t0\n"
        "print('PERFJSON ' + json.dumps({{'solve_s': dt, "
        "'count': int(app.solver.count)}}))\n"
    ).format(
        mod=EXAMPLE,
        argv=['--dx', str(dx), '-d', str(outdir), '--max-steps', str(steps),
              '--pfreq', '1000000'],
    )
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True)
    wall = time.perf_counter() - t0
    perf = None
    for line in proc.stdout.splitlines():
        if line.startswith('PERFJSON '):
            perf = json.loads(line[len('PERFJSON '):])
    if perf is None:
        sys.stderr.write(proc.stdout[-3000:] + '\n' + proc.stderr[-3000:])
        raise SystemExit('CPU run did not report PERFJSON')
    perf['subprocess_wall_s'] = wall
    perf['s_per_step'] = perf['solve_s'] / max(perf['count'], 1)
    return perf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=0.011,
                   help='~1.07M particles at 0.011.')
    p.add_argument('--cpu-steps', type=int, default=20)
    p.add_argument('--warp-perf-steps', type=int, default=20)
    p.add_argument('--snapshot-tf', type=float, default=0.20,
                   help='Continue Warp to this physical time for the snapshot '
                        '(0 = perf only, no developed snapshot).')
    p.add_argument('--hdx', type=float, default=1.3)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--n-damp', type=int, default=50)
    p.add_argument('--prefix', default='bench-1M')
    args = p.parse_args()

    out_dir = Path(__file__).parent
    import warp as wp

    # --- Warp IC (also gives the particle count) ---
    runner = WarpDamBreak3DRunner(
        dx=args.dx, hdx=args.hdx, c0=args.c0, cfl=args.cfl,
        n_damp=args.n_damp, adaptive_dt=True,
    )
    fluid, wall = runner.create_particles()
    n_fluid = fluid.get_number_of_particles()
    n_wall = wall.get_number_of_particles()
    n_total = n_fluid + n_wall
    print('[bench] particles: fluid=%d wall=%d total=%d (dx=%.4f)'
          % (n_fluid, n_wall, n_total, args.dx), flush=True)
    nnps = UniformGridWarpNNPS(
        dim=3, particles=[fluid, wall], radius_scale=runner.radius_scale
    )

    def warp_step(n_now, push):
        scale = damp_factor(n_now, runner.n_damp)
        return wc_sph_dam_break_step(
            nnps, fluid_index=0, solid_indices=(1,), dt=runner.dt,
            rho0=runner.rho0, c0=runner.c0, gamma=runner.gamma,
            alpha=runner.alpha, beta=runner.beta, kernel='wendland',
            xsph_eps=runner.xsph_eps, gz=runner.gz, gravity_ramp=1.0,
            adaptive_dt=True, cfl=runner.cfl, dt_min=runner.dt_min,
            dt_max=runner.dt_max, adaptive_dt_scale=scale,
            step_dt_max=runner.dt_max, push=push, return_dt=True,
        )

    # --- Warp per-step throughput (fixed short count) ---
    t_sim, count = 0.0, 0
    t_sim += warp_step(count, push=True); count += 1   # warmup (module load)
    wp.synchronize_device(nnps.device)
    t0 = time.perf_counter()
    for _ in range(args.warp_perf_steps):
        t_sim += warp_step(count, push=False); count += 1
    wp.synchronize_device(nnps.device)
    warp_per_step = (time.perf_counter() - t0) / args.warp_perf_steps

    # --- CPU per-step throughput (subprocess, fixed short count) ---
    tmp = tempfile.mkdtemp(prefix='dam_break_1M_', dir=str(out_dir))
    cpu = cpu_per_step(args.dx, args.cpu_steps, tmp)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    perf = {
        'case': 'Lobovsky 3D dam-break, no obstacle (~1M particles)',
        'dx': args.dx, 'c0': args.c0, 'kernel': 'wendland',
        'n_fluid': n_fluid, 'n_wall': n_wall, 'n_total': n_total,
        'cpu_pysph_fp64': {
            's_per_step': cpu['s_per_step'], 'steps': cpu['count'],
            'solve_s': cpu['solve_s'],
            'particle_steps_per_s': n_total / cpu['s_per_step'],
        },
        'warp_gpu_fp32': {
            's_per_step': warp_per_step, 'steps': args.warp_perf_steps,
            'particle_steps_per_s': n_total / warp_per_step,
        },
        'speedup_per_step': cpu['s_per_step'] / warp_per_step,
        'device': 'RTX 4060 Laptop (fp32) vs single-thread PySPH Cython (fp64)',
    }
    print('PERF ' + json.dumps(perf, indent=2, sort_keys=True), flush=True)
    (out_dir / f'{args.prefix}-perf.json').write_text(
        json.dumps(perf, indent=2, sort_keys=True))
    print('[bench] perf written; per-step speedup %.1fx (cpu %.3fs warp %.3fs)'
          % (perf['speedup_per_step'], cpu['s_per_step'], warp_per_step),
          flush=True)

    if args.snapshot_tf <= 0:
        return 0

    # --- continue Warp to a developed time for the 3D-explicit snapshot ---
    print('[bench] developing to tf=%.3f for snapshot (warp per-step ~%.3fs -> '
          '~%.0f steps)...' % (args.snapshot_tf, warp_per_step,
                               args.snapshot_tf / 9e-4), flush=True)
    while t_sim < args.snapshot_tf - 1e-12:
        t_sim += warp_step(count, push=False); count += 1
        if count % 200 == 0:
            print('[bench]   step %d  t=%.4f' % (count, t_sim), flush=True)
    fluid.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')
    wall.gpu.pull('x', 'y', 'z')
    fx, fy, fz = np.asarray(fluid.x), np.asarray(fluid.y), np.asarray(fluid.z)
    fu, fv, fw = np.asarray(fluid.u), np.asarray(fluid.v), np.asarray(fluid.w)
    wx, wy, wz = np.asarray(wall.x), np.asarray(wall.y), np.asarray(wall.z)
    speed = np.sqrt(fu*fu + fv*fv + fw*fw)
    vmax = max(float(speed.max()), 1e-6)

    def sub(n_pts, *arrs):
        if arrs[0].size <= n_pts:
            return arrs
        rng = np.linspace(0, arrs[0].size - 1, n_pts).astype(int)
        return tuple(a[rng] for a in arrs)

    fxp, fzp, fyp, sp = sub(200000, fx, fz, fy, speed)
    wxp, wzp, wyp = sub(120000, wx, wz, wy)
    fx3, fy3, fz3, s3 = sub(25000, fx, fy, fz, speed)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Warp GPU (fp32) 3D dam break, %d fluid + %d wall = %d total, '
                 't=%.3f s  (dx=%.3f, WendlandQuintic)'
                 % (n_fluid, n_wall, n_total, t_sim, args.dx), fontsize=12)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(wxp, wzp, s=2, c='0.8', marker='s', linewidths=0)
    s1 = ax1.scatter(fxp, fzp, c=sp, s=2, vmin=0, vmax=vmax, cmap='viridis',
                     linewidths=0)
    ax1.set_title('x-z side view (all y projected)')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('z (m)')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(s1, ax=ax1, label='speed (m/s)', shrink=0.8)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(wxp, wyp, s=2, c='0.85', marker='s', linewidths=0)
    s2 = ax2.scatter(fxp, fyp, c=sp, s=2, vmin=0, vmax=vmax, cmap='viridis',
                     linewidths=0)
    ax2.set_title('x-y top-down view (channel width in y)')
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(s2, ax=ax2, label='speed (m/s)', shrink=0.8)

    ax3 = fig.add_subplot(2, 2, 3)
    s3c = ax3.scatter(fyp, fzp, c=sp, s=2, vmin=0, vmax=vmax, cmap='viridis',
                      linewidths=0)
    ax3.set_title('y-z end view (cross-channel structure)')
    ax3.set_xlabel('y (m)'); ax3.set_ylabel('z (m)')
    ax3.set_aspect('equal', adjustable='box')
    fig.colorbar(s3c, ax=ax3, label='speed (m/s)', shrink=0.8)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(fx3, fy3, fz3, c=s3, s=1, vmin=0, vmax=vmax, cmap='viridis',
                linewidths=0)
    ax4.set_title('3D scatter (subsampled %d)' % fx3.size)
    ax4.set_xlabel('x'); ax4.set_ylabel('y'); ax4.set_zlabel('z')
    try:
        ax4.set_box_aspect((np.ptp(fx3), max(np.ptp(fy3), 1e-3), np.ptp(fz3)))
    except Exception:
        pass
    ax4.view_init(elev=18, azim=-72)

    img = out_dir / f'{args.prefix}-3d-snapshot.png'
    fig.savefig(img, dpi=140)
    plt.close(fig)
    perf['snapshot_image'] = str(img)
    perf['snapshot_t'] = t_sim
    (out_dir / f'{args.prefix}-perf.json').write_text(
        json.dumps(perf, indent=2, sort_keys=True))
    print('[bench] snapshot written: %s (t=%.3f, %d steps)'
          % (img, t_sim, count), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
