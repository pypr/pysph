#!/usr/bin/env python3
"""Tier-2 resolved parity: real PySPH dam-break Application vs the Warp step.

Runs the shipped CPU reference ``dam_break_3d_lobovsky.py`` (WCSPHScheme +
EPECIntegrator + WendlandQuintic) as a subprocess at a coarse resolution for a
short horizon, then:

1. loads the earliest dump (t ~ 0) and uses its *exact* fluid + boundary arrays
   as a shared initial condition (identical particle layout/ordering), and
2. advances the additive Warp ``wc_sph_dam_break_step`` from that IC, snapshotting
   at each CPU checkpoint time,

and reports, per checkpoint, signed CPU-vs-Warp deltas on aggregate observables
(kinetic energy, surge-front x, max fluid height, density/pressure ranges, wall
pressure) plus a short-horizon per-particle delta at the first checkpoint.

Per the ADR, the headline validators are the *aggregate* observables: fp32 (Warp)
vs fp64 (CPU) on a chaotic free-surface flow defeats long-horizon per-particle
parity, so the per-particle delta is reported for the earliest checkpoint only.
The Warp step is EPEC (matching the reference EPECIntegrator); its sound speed
``c0`` matches the reference scheme constant ``10*sqrt(2*9.81*0.55)``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_dam_break_step
from pysph.solver.utils import get_files, load

from dam_break_3d_runner import GRAVITY, H, REF_C0, damp_factor

EXAMPLE = 'pysph.examples.dam_break.dam_break_3d_lobovsky'


def _run_cpu_reference(dx, tf, pfreq, outdir):
    """Subprocess the real Application; run() only (skips mayavi post_process)."""
    code = (
        "from {mod} import DamBreak3D; "
        "DamBreak3D().run(argv={argv!r})"
    ).format(
        mod=EXAMPLE,
        argv=['--dx', str(dx), '-d', str(outdir), '--tf', str(tf),
              '--pfreq', str(pfreq), '--detailed-output'],
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def _load_sorted(outdir):
    files = get_files(str(outdir))
    snaps = []
    for f in files:
        data = load(f)
        t = float(data['solver_data']['t'])
        snaps.append((t, data['arrays']))
    snaps.sort(key=lambda s: s[0])
    return snaps


def _fluid_wall(arrays):
    """Return (fluid, boundary) ParticleArrays regardless of key naming."""
    fluid = arrays['fluid']
    wall = arrays['boundary'] if 'boundary' in arrays else arrays['wall']
    return fluid, wall


def _warp_from(pa, name, c0, rho0):
    n = pa.get_number_of_particles()
    z = np.zeros(n)
    rho = np.asarray(pa.rho, dtype=np.float64).copy() if 'rho' in pa.properties \
        else np.ones(n) * rho0
    return get_particle_array(
        name=name,
        x=np.asarray(pa.x, dtype=np.float64).copy(),
        y=np.asarray(pa.y, dtype=np.float64).copy(),
        z=np.asarray(pa.z, dtype=np.float64).copy(),
        h=np.asarray(pa.h, dtype=np.float64).copy(),
        m=np.asarray(pa.m, dtype=np.float64).copy(),
        rho=rho, p=z.copy(), cs=np.ones(n) * c0,
        u=z.copy(), v=z.copy(), w=z.copy(),
        au=z.copy(), av=z.copy(), aw=z.copy(), arho=z.copy(),
        ax=z.copy(), ay=z.copy(), az=z.copy(),
        x0=z.copy(), y0=z.copy(), z0=z.copy(),
        u0=z.copy(), v0=z.copy(), w0=z.copy(), rho0=z.copy(),
        backend='warp',
    )


def _observables(fx, fy, fz, fu, fv, fw, frho, fp, fm, wp_):
    speed2 = fu*fu + fv*fv + fw*fw
    return {
        'kinetic_energy': float(0.5 * np.sum(fm * speed2)),
        'surge_front_x': float(np.max(fx)),
        'max_height': float(np.max(fz)),
        'min_z': float(np.min(fz)),
        'rho_min': float(np.min(frho)),
        'rho_max': float(np.max(frho)),
        'p_min': float(np.min(fp)),
        'p_max': float(np.max(fp)),
        'wall_p_max': float(np.max(wp_)),
        'mean_speed': float(np.mean(np.sqrt(speed2))),
    }


def _cpu_observables(fluid, wall):
    def prop(pa, name, default=0.0):
        if name in pa.properties:
            return np.asarray(getattr(pa, name), dtype=np.float64)
        return np.full(pa.get_number_of_particles(), default)
    return _observables(
        prop(fluid, 'x'), prop(fluid, 'y'), prop(fluid, 'z'),
        prop(fluid, 'u'), prop(fluid, 'v'), prop(fluid, 'w'),
        prop(fluid, 'rho', 1000.0), prop(fluid, 'p'), prop(fluid, 'm'),
        prop(wall, 'p'),
    )


def _warp_observables(fluid, wall):
    return _observables(
        fluid.x, fluid.y, fluid.z, fluid.u, fluid.v, fluid.w,
        fluid.rho, fluid.p, fluid.m, wall.p,
    )


def _step_to(nnps, fluid_idx, wall_idx, t_target, t_now, count, args, push):
    """Advance Warp until time reaches t_target; return (t_now, count)."""
    first = push
    while t_now < t_target - 1e-12:
        scale = damp_factor(count, args.n_damp)
        remaining = t_target - t_now
        dt_used = wc_sph_dam_break_step(
            nnps, fluid_index=fluid_idx, solid_indices=(wall_idx,),
            dt=args.dt_max, rho0=args.rho0, c0=args.c0, gamma=args.gamma,
            alpha=args.alpha, beta=args.beta, kernel='wendland',
            xsph_eps=args.xsph_eps, gz=args.gz, gravity_ramp=1.0,
            adaptive_dt=True, cfl=args.cfl, dt_min=args.dt_min,
            dt_max=args.dt_max, adaptive_dt_scale=scale,
            step_dt_max=min(args.dt_max, remaining), push=first,
            return_dt=True,
        )
        first = False
        t_now += dt_used
        count += 1
    return t_now, count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=0.1)
    p.add_argument('--tf', type=float, default=0.06)
    p.add_argument('--pfreq', type=int, default=20)
    p.add_argument('--rho0', type=float, default=1000.0)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--gamma', type=float, default=7.0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--xsph-eps', type=float, default=0.5)
    p.add_argument('--gz', type=float, default=-GRAVITY)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--n-damp', type=int, default=50)
    p.add_argument('--dt-min', type=float, default=0.0)
    p.add_argument('--dt-max', type=float, default=None)
    p.add_argument('--radius-scale', type=float, default=2.0)
    p.add_argument('--keep-output', action='store_true')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--prefix', default='comparison-resolved')
    args = p.parse_args()

    if args.dt_max is None:
        h0 = 1.3 * args.dx
        co = 10.0 * np.sqrt(2.0 * GRAVITY * H)
        args.dt_max = 0.25 * h0 / (1.1 * co)

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix='dam_break_cpu_', dir=str(out_dir))
    _run_cpu_reference(args.dx, args.tf, args.pfreq, tmp)
    snaps = _load_sorted(tmp)
    if len(snaps) < 2:
        raise SystemExit("Reference produced <2 snapshots; raise --tf/--pfreq")

    t0, ic = snaps[0]
    fluid_ic, wall_ic = _fluid_wall(ic)
    fluid = _warp_from(fluid_ic, 'fluid', args.c0, args.rho0)
    wall = _warp_from(wall_ic, 'wall', args.c0, args.rho0)
    nnps = UniformGridWarpNNPS(
        dim=3, particles=[fluid, wall], radius_scale=args.radius_scale
    )

    checkpoints = []
    t_now, count, push = t0, 0, True
    for k, (t_cp, arrays) in enumerate(snaps[1:]):
        t_now, count = _step_to(
            nnps, 0, 1, t_cp, t_now, count, args, push
        )
        push = False
        fluid.gpu.pull('x', 'y', 'z', 'rho', 'p', 'u', 'v', 'w')
        wall.gpu.pull('x', 'y', 'z', 'rho', 'p')
        cpu_fluid, cpu_wall = _fluid_wall(arrays)
        cpu_obs = _cpu_observables(cpu_fluid, cpu_wall)
        warp_obs = _warp_observables(fluid, wall)
        deltas = {key: warp_obs[key] - cpu_obs[key] for key in cpu_obs}
        finite = bool(
            np.all(np.isfinite(fluid.x)) and np.all(np.isfinite(fluid.rho))
            and np.all(np.isfinite(fluid.p))
        )
        entry = {
            'checkpoint': k,
            'cpu_t': t_cp,
            'warp_t': t_now,
            'cpu': cpu_obs,
            'warp': warp_obs,
            'delta_warp_minus_cpu': deltas,
            'warp_all_finite': finite,
        }
        # Short-horizon per-particle delta at the first checkpoint only.
        if k == 0 and fluid.get_number_of_particles() == \
                cpu_fluid.get_number_of_particles():
            entry['per_particle_max_abs'] = {
                'x': float(np.max(np.abs(fluid.x - np.asarray(cpu_fluid.x)))),
                'z': float(np.max(np.abs(fluid.z - np.asarray(cpu_fluid.z)))),
                'rho': float(np.max(np.abs(
                    fluid.rho - np.asarray(cpu_fluid.rho)))),
            }
        checkpoints.append(entry)

    report = {
        'params': vars(args),
        'reference_example': EXAMPLE,
        'n_snapshots': len(snaps),
        'ic_t': t0,
        'fluid_particles': int(fluid.get_number_of_particles()),
        'wall_particles': int(wall.get_number_of_particles()),
        'checkpoints': checkpoints,
        'all_finite': all(c['warp_all_finite'] for c in checkpoints),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    summary_path = out_dir / f'{args.prefix}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    if not args.keep_output:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    if not report['all_finite']:
        raise SystemExit("Warp produced non-finite values at a checkpoint")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
