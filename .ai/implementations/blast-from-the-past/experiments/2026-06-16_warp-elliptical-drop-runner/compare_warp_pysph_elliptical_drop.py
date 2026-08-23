#!/usr/bin/env python3
"""Compare Warp elliptical drop against a CPU PySPH-primitive baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from cyarray.carray import UIntArray

from pysph.base.kernels import Gaussian
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array

from warp_elliptical_drop_runner import WarpEllipticalDropRunner


def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return nbrs.get_npy_array()[:nbrs.length]


def _create_cpu_particles(nx, rho0, hdx):
    dx = 1.0 / nx
    x, y = np.mgrid[-1.05:1.05 + 1.0e-4:dx,
                    -1.05:1.05 + 1.0e-4:dx]
    condition = ~((x*x + y*y - 1.0) > 1.0e-10)
    x = np.asarray(x[condition].ravel(), dtype=np.float64)
    y = np.asarray(y[condition].ravel(), dtype=np.float64)
    z = np.zeros_like(x)
    m = np.ones_like(x) * dx * dx * rho0
    h = np.ones_like(x) * hdx * dx
    rho = np.ones_like(x) * rho0
    p = np.zeros_like(x)
    cs = np.zeros_like(x)
    u = -100.0 * x
    v = 100.0 * y
    w = np.zeros_like(x)
    zeros = np.zeros_like(x)
    pa = get_particle_array(
        name='fluid', x=x, y=y, z=z, h=h, m=m, rho=rho, p=p, cs=cs,
        u=u, v=v, w=w, au=zeros.copy(), av=zeros.copy(), aw=zeros.copy(),
        ax=zeros.copy(), ay=zeros.copy(), az=zeros.copy(),
        dt_cfl=zeros.copy(), dt_force=zeros.copy(),
    )
    return pa


def _summation_density(pa, kernel, radius_scale):
    nnps = LinkedListNNPS(dim=2, particles=[pa], radius_scale=radius_scale)
    rho = np.zeros_like(pa.rho)
    for i in range(pa.get_number_of_particles()):
        total = 0.0
        for j in _neighbors(nnps, 0, 0, i):
            xij = [pa.x[i] - pa.x[j], pa.y[i] - pa.y[j], 0.0]
            rij = np.sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            hij = 0.5 * (pa.h[i] + pa.h[j])
            total += pa.m[j] * kernel.kernel(xij=xij, rij=rij, h=hij)
        rho[i] = total
    pa.rho[:] = rho


def _tait_eos(pa, rho0, c0, gamma, p0):
    ratio = pa.rho / rho0
    pa.p[:] = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1.0)
    pa.cs[:] = c0 * ratio**(0.5 * (gamma - 1.0))


def _pressure_gradient(pa, kernel, radius_scale):
    nnps = LinkedListNNPS(dim=2, particles=[pa], radius_scale=radius_scale)
    au = np.zeros_like(pa.au)
    av = np.zeros_like(pa.av)
    aw = np.zeros_like(pa.aw)
    for i in range(pa.get_number_of_particles()):
        rhoi21 = 1.0 / (pa.rho[i] * pa.rho[i])
        tmpi = pa.p[i] * rhoi21
        for j in _neighbors(nnps, 0, 0, i):
            xij = [pa.x[i] - pa.x[j], pa.y[i] - pa.y[j], 0.0]
            rij = np.sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            hij = 0.5 * (pa.h[i] + pa.h[j])
            dwij = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            rhoj21 = 1.0 / (pa.rho[j] * pa.rho[j])
            fac = -pa.m[j] * (tmpi + pa.p[j] * rhoj21)
            au[i] += fac * dwij[0]
            av[i] += fac * dwij[1]
            aw[i] += fac * dwij[2]
    pa.au[:] = au
    pa.av[:] = av
    pa.aw[:] = aw


def _artificial_viscosity(pa, kernel, radius_scale, alpha, beta):
    nnps = LinkedListNNPS(dim=2, particles=[pa], radius_scale=radius_scale)
    for i in range(pa.get_number_of_particles()):
        for j in _neighbors(nnps, 0, 0, i):
            xij = [pa.x[i] - pa.x[j], pa.y[i] - pa.y[j], 0.0]
            vij = [pa.u[i] - pa.u[j], pa.v[i] - pa.v[j], 0.0]
            vdotx = vij[0]*xij[0] + vij[1]*xij[1]
            if vdotx < 0.0:
                rij2 = xij[0]*xij[0] + xij[1]*xij[1]
                rij = np.sqrt(rij2)
                hij = 0.5 * (pa.h[i] + pa.h[j])
                mu = hij * vdotx / (rij2 + 0.01*hij*hij)
                rhoij1 = 2.0 / (pa.rho[i] + pa.rho[j])
                cij = 0.5 * (pa.cs[i] + pa.cs[j])
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                dwij = [0.0, 0.0, 0.0]
                kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
                fac = -pa.m[j] * piij
                pa.au[i] += fac * dwij[0]
                pa.av[i] += fac * dwij[1]
                pa.aw[i] += fac * dwij[2]


def _xsph(pa, kernel, radius_scale, eps):
    pa.ax[:] = 0.0
    pa.ay[:] = 0.0
    pa.az[:] = 0.0
    if eps is None or eps == 0.0:
        return
    nnps = LinkedListNNPS(dim=2, particles=[pa], radius_scale=radius_scale)
    for i in range(pa.get_number_of_particles()):
        for j in _neighbors(nnps, 0, 0, i):
            xij = [pa.x[i] - pa.x[j], pa.y[i] - pa.y[j], 0.0]
            vij = [pa.u[i] - pa.u[j], pa.v[i] - pa.v[j], 0.0]
            rij = np.sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            hij = 0.5 * (pa.h[i] + pa.h[j])
            wij = kernel.kernel(xij=xij, rij=rij, h=hij)
            rhoij1 = 2.0 / (pa.rho[i] + pa.rho[j])
            tmp = -eps * pa.m[j] * wij * rhoij1
            pa.ax[i] += tmp * vij[0]
            pa.ay[i] += tmp * vij[1]


def _adaptive_dt(pa, radius_scale, c0, cfl, dt_min, dt_max):
    nnps = LinkedListNNPS(dim=2, particles=[pa], radius_scale=radius_scale)
    pa.dt_cfl[:] = 0.0
    pa.dt_force[:] = pa.au*pa.au + pa.av*pa.av + pa.aw*pa.aw
    for i in range(pa.get_number_of_particles()):
        for j in _neighbors(nnps, 0, 0, i):
            xij = [pa.x[i] - pa.x[j], pa.y[i] - pa.y[j], 0.0]
            vij = [pa.u[i] - pa.u[j], pa.v[i] - pa.v[j], 0.0]
            rij2 = xij[0]*xij[0] + xij[1]*xij[1]
            if rij2 > 1.0e-12:
                hij = 0.5 * (pa.h[i] + pa.h[j])
                vdotx = vij[0]*xij[0] + vij[1]*xij[1]
                fac = abs(hij * vdotx / rij2) + c0
                pa.dt_cfl[i] = max(pa.dt_cfl[i], fac)
    hmin = np.min(pa.h)
    dt = dt_max
    max_cfl = np.max(pa.dt_cfl)
    max_force = np.max(pa.dt_force)
    if max_cfl > 0.0:
        dt = min(dt, cfl * hmin / max_cfl)
    if max_force > 0.0:
        dt = min(dt, cfl * np.sqrt(hmin / np.sqrt(max_force)))
    return min(max(dt, dt_min), dt_max)


def _compute_acceleration(pa, kernel, radius_scale, rho0, c0, p0, gamma,
                          alpha, beta):
    _summation_density(pa, kernel, radius_scale)
    _tait_eos(pa, rho0, c0, gamma, p0)
    _pressure_gradient(pa, kernel, radius_scale)
    if alpha != 0.0 or beta != 0.0:
        _artificial_viscosity(pa, kernel, radius_scale, alpha, beta)


def _run_cpu(args):
    pa = _create_cpu_particles(args.nx, args.rho0, args.hdx)
    kernel = Gaussian(dim=2)
    radius_scale = 3.0
    dt_history = []
    time = 0.0
    for _ in range(args.steps):
        _compute_acceleration(
            pa, kernel, radius_scale, args.rho0, args.c0, args.p0,
            args.gamma, args.alpha, args.beta
        )
        if args.adaptive_dt:
            dt = _adaptive_dt(
                pa, radius_scale, args.c0, args.cfl, args.dt_min,
                args.dt_max
            )
        else:
            dt = args.dt
        pa.u[:] += 0.5 * dt * pa.au
        pa.v[:] += 0.5 * dt * pa.av
        pa.w[:] += 0.5 * dt * pa.aw
        _xsph(pa, kernel, radius_scale, args.xsph_eps)
        pa.x[:] += dt * (pa.u + pa.ax)
        pa.y[:] += dt * (pa.v + pa.ay)
        pa.z[:] += dt * (pa.w + pa.az)
        _compute_acceleration(
            pa, kernel, radius_scale, args.rho0, args.c0, args.p0,
            args.gamma, args.alpha, args.beta
        )
        pa.u[:] += 0.5 * dt * pa.au
        pa.v[:] += 0.5 * dt * pa.av
        pa.w[:] += 0.5 * dt * pa.aw
        dt_history.append(dt)
        time += dt
    return pa, np.asarray(dt_history), time


def _metrics(pa, dt_history, time):
    speed2 = pa.u*pa.u + pa.v*pa.v + pa.w*pa.w
    radius = np.sqrt(pa.x*pa.x + pa.y*pa.y)
    return {
        'particles': int(pa.get_number_of_particles()),
        'time': float(time),
        'dt_min_used': float(np.min(dt_history)),
        'dt_max_used': float(np.max(dt_history)),
        'rho_min': float(np.min(pa.rho)),
        'rho_max': float(np.max(pa.rho)),
        'radius_max': float(np.max(radius)),
        'kinetic_energy': float(0.5 * np.sum(pa.m * speed2)),
        'all_finite': bool(all(np.all(np.isfinite(getattr(pa, name)))
                               for name in ('x', 'y', 'rho', 'p', 'u', 'v'))),
    }


def _save_cpu(path, pa, dt_history, metrics):
    np.savez(
        path, x=pa.x, y=pa.y, z=pa.z, h=pa.h, m=pa.m, rho=pa.rho, p=pa.p,
        cs=pa.cs, u=pa.u, v=pa.v, w=pa.w, au=pa.au, av=pa.av, aw=pa.aw,
        ax=pa.ax, ay=pa.ay, az=pa.az, dt_cfl=pa.dt_cfl,
        dt_force=pa.dt_force, dt_history=dt_history,
        metrics=json.dumps(metrics, sort_keys=True),
    )


def _plot_side_by_side(cpu_path, warp_path, image_path):
    cpu = np.load(cpu_path)
    warp = np.load(warp_path)
    cpu_speed = np.sqrt(cpu['u']*cpu['u'] + cpu['v']*cpu['v'])
    warp_speed = np.sqrt(warp['u']*warp['u'] + warp['v']*warp['v'])
    vmax = max(float(cpu_speed.max()), float(warp_speed.max()))
    xmin = min(float(cpu['x'].min()), float(warp['x'].min()))
    xmax = max(float(cpu['x'].max()), float(warp['x'].max()))
    ymin = min(float(cpu['y'].min()), float(warp['y'].min()))
    ymax = max(float(cpu['y'].max()), float(warp['y'].max()))
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, data, speed, title in (
        (axes[0], cpu, cpu_speed, 'CPU PySPH baseline'),
        (axes[1], warp, warp_speed, 'Warp GPU'),
    ):
        sc = ax.scatter(data['x'], data['y'], c=speed, s=8, vmin=0.0,
                        vmax=vmax, cmap='viridis')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.colorbar(sc, ax=axes, label='speed')
    fig.savefig(image_path, dpi=180)
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--dt', type=float, default=1.0e-5)
    parser.add_argument('--rho0', type=float, default=1.0)
    parser.add_argument('--c0', type=float, default=20.0)
    parser.add_argument('--p0', type=float, default=0.0)
    parser.add_argument('--hdx', type=float, default=1.3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=7.0)
    parser.add_argument('--xsph-eps', type=float, default=0.5)
    parser.add_argument('--adaptive-dt', action='store_true')
    parser.add_argument('--cfl', type=float, default=0.25)
    parser.add_argument('--dt-min', type=float, default=1.0e-7)
    parser.add_argument('--dt-max', type=float, default=1.0e-5)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--prefix', default='comparison-smoke')
    return parser.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_path = out_dir / f'{args.prefix}-cpu.npz'
    warp_path = out_dir / f'{args.prefix}-warp.npz'
    image_path = out_dir / f'{args.prefix}.png'

    runner = WarpEllipticalDropRunner(
        nx=args.nx, steps=args.steps, dt=args.dt, rho0=args.rho0,
        c0=args.c0, p0=args.p0, hdx=args.hdx, alpha=args.alpha,
        beta=args.beta, eos='tait', gamma=args.gamma, kernel='gaussian',
        xsph_eps=args.xsph_eps, adaptive_dt=args.adaptive_dt, cfl=args.cfl,
        dt_min=args.dt_min, dt_max=args.dt_max, output=warp_path
    )
    warp_metrics = runner.run()
    cpu_pa, cpu_dt_history, cpu_time = _run_cpu(args)
    cpu_metrics = _metrics(cpu_pa, cpu_dt_history, cpu_time)
    _save_cpu(cpu_path, cpu_pa, cpu_dt_history, cpu_metrics)
    _plot_side_by_side(cpu_path, warp_path, image_path)

    metrics = {
        'cpu': cpu_metrics,
        'warp': warp_metrics,
        'cpu_output': str(cpu_path),
        'warp_output': str(warp_path),
        'image': str(image_path),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not cpu_metrics['all_finite']:
        raise SystemExit("CPU baseline produced non-finite values")
    if not warp_metrics['all_finite']:
        raise SystemExit("Warp run produced non-finite values")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
