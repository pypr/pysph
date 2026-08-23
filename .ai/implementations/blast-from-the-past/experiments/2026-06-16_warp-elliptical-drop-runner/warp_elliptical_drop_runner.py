#!/usr/bin/env python3
"""Application-style Warp runner for the elliptical-drop initial condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_leapfrog_step


class WarpEllipticalDropRunner:
    """Small runner around the current Warp WCSPH prototype.

    This intentionally does not use PySPH's Application/Solver stack yet. It
    gives the Warp kernels a repeatable elliptical-drop-style workload while
    the missing WCSPH terms are still being ported.
    """

    def __init__(self, nx=8, steps=2, dt=1.0e-5, rho0=1.0, c0=20.0,
                 p0=0.0, hdx=1.3, alpha=0.1, beta=0.0, eos='tait',
                 gamma=7.0, kernel='gaussian', radius_scale=None,
                 xsph_eps=0.5, adaptive_dt=False, cfl=0.25, dt_min=0.0,
                 dt_max=None, density_mode='summation', output=None):
        self.nx = int(nx)
        self.steps = int(steps)
        self.dt = float(dt)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.p0 = float(p0)
        self.hdx = float(hdx)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eos = eos
        self.gamma = float(gamma)
        self.kernel = kernel
        self.radius_scale = (
            3.0 if kernel == 'gaussian' and radius_scale is None
            else 2.0 if radius_scale is None
            else float(radius_scale)
        )
        self.xsph_eps = None if xsph_eps is None else float(xsph_eps)
        self.adaptive_dt = bool(adaptive_dt)
        self.cfl = float(cfl)
        self.dt_min = float(dt_min)
        self.dt_max = self.dt if dt_max is None else float(dt_max)
        self.density_mode = density_mode
        self.dx = 1.0 / self.nx
        self.output = Path(output) if output is not None else None
        self.dt_history = []

    def create_particles(self):
        dx = self.dx
        x, y = np.mgrid[-1.05:1.05 + 1.0e-4:dx,
                        -1.05:1.05 + 1.0e-4:dx]
        condition = ~((x*x + y*y - 1.0) > 1.0e-10)
        x = np.asarray(x[condition].ravel(), dtype=np.float64)
        y = np.asarray(y[condition].ravel(), dtype=np.float64)
        z = np.zeros_like(x)

        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        rho = np.ones_like(x) * self.rho0
        p = np.zeros_like(x)
        cs = np.ones_like(x) * self.c0
        u = -100.0 * x
        v = 100.0 * y
        w = np.zeros_like(x)
        au = np.zeros_like(x)
        av = np.zeros_like(x)
        aw = np.zeros_like(x)
        arho = np.zeros_like(x)
        ax = np.zeros_like(x)
        ay = np.zeros_like(x)
        az = np.zeros_like(x)
        x0 = np.zeros_like(x)
        y0 = np.zeros_like(x)
        z0 = np.zeros_like(x)
        u0 = np.zeros_like(x)
        v0 = np.zeros_like(x)
        w0 = np.zeros_like(x)
        rho_ref = np.zeros_like(x)

        return get_particle_array(
            name='fluid', x=x, y=y, z=z, h=h, m=m, rho=rho, p=p,
            cs=cs, u=u, v=v, w=w, au=au, av=av, aw=aw, arho=arho,
            ax=ax, ay=ay, az=az, x0=x0, y0=y0, z0=z0, u0=u0, v0=v0,
            w0=w0, rho0=rho_ref, backend='warp'
        )

    def run(self):
        pa = self.create_particles()
        nnps = UniformGridWarpNNPS(
            dim=2, particles=[pa], radius_scale=self.radius_scale
        )

        time = 0.0
        for _ in range(self.steps):
            _, dt_used = wc_sph_leapfrog_step(
                nnps, dt=self.dt, rho0=self.rho0, c0=self.c0, p0=self.p0,
                alpha=self.alpha, beta=self.beta, eos=self.eos,
                gamma=self.gamma, kernel=self.kernel, xsph_eps=self.xsph_eps,
                adaptive_dt=self.adaptive_dt, cfl=self.cfl,
                dt_min=self.dt_min, dt_max=self.dt_max, return_dt=True,
                density_mode=self.density_mode
            )
            self.dt_history.append(dt_used)
            time += dt_used

        pull_props = [
            'x', 'y', 'z', 'rho', 'p', 'cs', 'u', 'v', 'w', 'au', 'av', 'aw'
        ]
        for optional in ('ax', 'ay', 'az', 'arho', 'dt_cfl', 'dt_force'):
            if optional in pa.properties:
                pull_props.append(optional)
        pa.gpu.pull(*pull_props)
        metrics = self._metrics(pa, time)
        if self.output is not None:
            self._write_output(pa, metrics)
        return metrics

    def _metrics(self, pa, time):
        finite_props = [
            'x', 'y', 'z', 'rho', 'p', 'cs', 'u', 'v', 'w', 'au', 'av', 'aw'
        ]
        finite_props.extend(
            name for name in ('ax', 'ay', 'az', 'arho', 'dt_cfl', 'dt_force')
            if name in pa.properties
        )
        finite = all(np.all(np.isfinite(getattr(pa, name)))
                     for name in finite_props)
        ke = 0.5 * np.sum(pa.m * (pa.u*pa.u + pa.v*pa.v + pa.w*pa.w))
        radius = np.sqrt(pa.x*pa.x + pa.y*pa.y)
        dt_history = np.asarray(self.dt_history)
        return {
            'particles': int(pa.get_number_of_particles()),
            'steps': self.steps,
            'dt': self.dt,
            'dt_min_used': float(np.min(dt_history)),
            'dt_max_used': float(np.max(dt_history)),
            'dt_last': float(dt_history[-1]),
            'time': float(time),
            'nx': self.nx,
            'rho_min': float(np.min(pa.rho)),
            'rho_max': float(np.max(pa.rho)),
            'c0': self.c0,
            'eos': self.eos,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'kernel': self.kernel,
            'radius_scale': self.radius_scale,
            'xsph_eps': self.xsph_eps,
            'adaptive_dt': self.adaptive_dt,
            'density_mode': self.density_mode,
            'cfl': self.cfl,
            'dt_min': self.dt_min,
            'dt_max': self.dt_max,
            'p_min': float(np.min(pa.p)),
            'p_max': float(np.max(pa.p)),
            'cs_min': float(np.min(pa.cs)),
            'cs_max': float(np.max(pa.cs)),
            'x_min': float(np.min(pa.x)),
            'x_max': float(np.max(pa.x)),
            'y_min': float(np.min(pa.y)),
            'y_max': float(np.max(pa.y)),
            'radius_max': float(np.max(radius)),
            'kinetic_energy': float(ke),
            'all_finite': bool(finite),
        }

    def _write_output(self, pa, metrics):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        optional = {}
        for name in ('ax', 'ay', 'az', 'arho', 'dt_cfl', 'dt_force'):
            if name in pa.properties:
                optional[name] = getattr(pa, name)
        np.savez(
            self.output,
            x=pa.x, y=pa.y, z=pa.z, h=pa.h, m=pa.m, rho=pa.rho, p=pa.p,
            cs=pa.cs, u=pa.u, v=pa.v, w=pa.w, au=pa.au, av=pa.av, aw=pa.aw,
            dt_history=np.asarray(self.dt_history),
            **optional,
            metrics=json.dumps(metrics, sort_keys=True),
        )


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
    parser.add_argument('--eos', choices=('isothermal', 'tait'),
                        default='tait')
    parser.add_argument('--gamma', type=float, default=7.0)
    parser.add_argument('--kernel', choices=('cubic', 'gaussian'),
                        default='gaussian')
    parser.add_argument('--radius-scale', type=float, default=None)
    parser.add_argument('--xsph-eps', type=float, default=0.5)
    parser.add_argument('--no-xsph', action='store_true')
    parser.add_argument('--adaptive-dt', action='store_true')
    parser.add_argument('--cfl', type=float, default=0.25)
    parser.add_argument('--dt-min', type=float, default=0.0)
    parser.add_argument('--dt-max', type=float, default=None)
    parser.add_argument('--density-mode', choices=('summation', 'continuity'),
                        default='summation')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    runner = WarpEllipticalDropRunner(
        nx=args.nx, steps=args.steps, dt=args.dt, rho0=args.rho0,
        c0=args.c0, p0=args.p0, hdx=args.hdx, alpha=args.alpha,
        beta=args.beta, eos=args.eos, gamma=args.gamma, kernel=args.kernel,
        radius_scale=args.radius_scale,
        xsph_eps=None if args.no_xsph else args.xsph_eps,
        adaptive_dt=args.adaptive_dt, cfl=args.cfl, dt_min=args.dt_min,
        dt_max=args.dt_max, density_mode=args.density_mode,
        output=args.output
    )
    metrics = runner.run()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics['particles'] <= 0:
        raise SystemExit("No particles were created")
    if not metrics['all_finite']:
        raise SystemExit("Non-finite values in final state")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
