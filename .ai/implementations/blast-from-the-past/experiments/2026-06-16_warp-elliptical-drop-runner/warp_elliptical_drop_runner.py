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
                 gamma=7.0, output=None):
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
        self.dx = 1.0 / self.nx
        self.output = Path(output) if output is not None else None

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

        return get_particle_array(
            name='fluid', x=x, y=y, z=z, h=h, m=m, rho=rho, p=p,
            cs=cs, u=u, v=v, w=w, au=au, av=av, aw=aw, backend='warp'
        )

    def run(self):
        pa = self.create_particles()
        nnps = UniformGridWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)

        for _ in range(self.steps):
            wc_sph_leapfrog_step(
                nnps, dt=self.dt, rho0=self.rho0, c0=self.c0, p0=self.p0,
                alpha=self.alpha, beta=self.beta, eos=self.eos,
                gamma=self.gamma
            )

        pa.gpu.pull('x', 'y', 'z', 'rho', 'p', 'cs', 'u', 'v', 'w', 'au', 'av',
                    'aw')
        metrics = self._metrics(pa)
        if self.output is not None:
            self._write_output(pa, metrics)
        return metrics

    def _metrics(self, pa):
        finite = all(
            np.all(np.isfinite(getattr(pa, name)))
            for name in ('x', 'y', 'z', 'rho', 'p', 'cs', 'u', 'v', 'w', 'au',
                         'av', 'aw')
        )
        ke = 0.5 * np.sum(pa.m * (pa.u*pa.u + pa.v*pa.v + pa.w*pa.w))
        radius = np.sqrt(pa.x*pa.x + pa.y*pa.y)
        return {
            'particles': int(pa.get_number_of_particles()),
            'steps': self.steps,
            'dt': self.dt,
            'time': self.steps * self.dt,
            'nx': self.nx,
            'rho_min': float(np.min(pa.rho)),
            'rho_max': float(np.max(pa.rho)),
            'c0': self.c0,
            'eos': self.eos,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
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
        np.savez(
            self.output,
            x=pa.x, y=pa.y, z=pa.z, h=pa.h, m=pa.m, rho=pa.rho, p=pa.p,
            cs=pa.cs, u=pa.u, v=pa.v, w=pa.w, au=pa.au, av=pa.av, aw=pa.aw,
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
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    runner = WarpEllipticalDropRunner(
        nx=args.nx, steps=args.steps, dt=args.dt, rho0=args.rho0,
        c0=args.c0, p0=args.p0, hdx=args.hdx, alpha=args.alpha,
        beta=args.beta, eos=args.eos, gamma=args.gamma, output=args.output
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
