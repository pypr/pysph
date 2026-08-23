#!/usr/bin/env python3
"""Application-style Warp runner for the 3D dam-break.

The default mirrors the PySPH ``dam_break_3d_lobovsky.py`` no-obstacle
reference (ADR-0005). ``--with-obstacle`` enables the fixed Kleefsman obstacle
already supported by the same ``DamBreak3DGeometry``. Both are advanced with
the additive Warp dam-break step::

    UniformGridWarpNNPS(dim=3, [fluid, wall(, obstacle)])
    wc_sph_dam_break_step      # EPEC, WendlandQuintic, Tait + Tait-HG walls

Physics parity notes (vs the reference scheme):

- ``c0`` defaults to the reference scheme's sound speed
  ``10*sqrt(2*9.81*0.55)`` (the module constant the WCSPHScheme is built with),
  *not* ``get_max_speed`` -- the reference uses the latter only for the initial
  ``dt`` guess. See the experiment.md for this known reference inconsistency.
- The Warp step is E-P-E-C (it re-evaluates accelerations before the predictor),
  which matches the reference ``EPECIntegrator``; gravity is full strength.
- The ``n_damp`` startup uses PySPH's *timestep* damping factor
  ``0.5*(sin(pi*(-0.5 + (count+1)/n_damp)) + 1)`` (Solver._damp_timestep),
  applied to the adaptive ``dt`` per step -- the reference does NOT ramp gravity
  (the additive ``gravity_ramp`` backend feature is left at 1.0 here).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_dam_break_step
from pysph.examples._db_geometry import DamBreak3DGeometry

# Lobovsky no-obstacle reference scale (dam_break_3d_lobovsky.py).
H = 1.0
GRAVITY = 9.81
# Reference scheme sound speed (module constant, built into the WCSPHScheme).
REF_C0 = 10.0 * math.sqrt(2.0 * GRAVITY * 0.55)


def damp_factor(count, n_damp):
    """PySPH ``Solver._damp_timestep`` factor for 0-based step ``count``."""
    if n_damp > 0 and count < n_damp:
        return 0.5 * (math.sin(math.pi * (-0.5 + (count + 1) / float(n_damp)))
                      + 1.0)
    return 1.0


class WarpDamBreak3DRunner:
    """Small Application-style runner around ``wc_sph_dam_break_step``.

    Like the elliptical-drop runner, this deliberately does not use PySPH's
    Application/Solver stack; it gives the additive 3D dam-break path a
    repeatable, reference-matched workload for correctness checks.
    """

    def __init__(self, dx=H / 15.0, hdx=1.3, steps=20, rho0=1000.0, c0=REF_C0,
                 p0=0.0, gamma=7.0, alpha=0.25, beta=0.0, kernel='wendland',
                 radius_scale=2.0, xsph_eps=0.5, gz=-GRAVITY, n_damp=50,
                 nboundary_layers=1, adaptive_dt=True, cfl=0.3, dt=None,
                 dt_min=0.0, dt_max=None, with_obstacle=False, output=None):
        self.dx = float(dx)
        self.hdx = float(hdx)
        self.steps = int(steps)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.p0 = float(p0)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kernel = kernel
        self.radius_scale = float(radius_scale)
        self.xsph_eps = None if xsph_eps is None else float(xsph_eps)
        self.gz = float(gz)
        self.n_damp = int(n_damp)
        self.nboundary_layers = int(nboundary_layers)
        self.adaptive_dt = bool(adaptive_dt)
        self.cfl = float(cfl)
        self.h0 = self.hdx * self.dx
        # Reference initial dt: 0.25*h0/(1.1*co), co = 10*get_max_speed.
        co = 10.0 * math.sqrt(2.0 * GRAVITY * H)
        ref_dt = 0.25 * self.h0 / (1.1 * co)
        self.dt = ref_dt if dt is None else float(dt)
        self.dt_min = float(dt_min)
        # PySPH treats ref_dt as the *initial/seed* dt only -- its adaptive
        # controller (Integrator.compute_time_step) then grows dt to the
        # CFL-limited value with NO clamp to the seed. So we must NOT cap the
        # adaptive dt at ref_dt; default to no cap (CFL + n_damp govern, exactly
        # like the reference). Capping at ref_dt would make Warp take ~1.7x more,
        # smaller steps than PySPH to the same physical time.
        self.dt_max = float('inf') if dt_max is None else float(dt_max)
        self.with_obstacle = bool(with_obstacle)
        self.output = Path(output) if output is not None else None
        self.dt_history = []
        self.geom = None

    def _build_geometry(self):
        return DamBreak3DGeometry(
            container_height=1.5 * H, container_width=H / 2.0,
            container_length=161 * H / 30.0, fluid_column_height=H,
            fluid_column_width=H / 2.0, fluid_column_length=2.0 * H,
            dx=self.dx, nboundary_layers=self.nboundary_layers,
            hdx=self.hdx, rho0=self.rho0,
            with_obstacle=self.with_obstacle,
        )

    def _to_warp(self, src, name):
        """Rebuild a CPU geometry array as a fully-propertied warp array."""
        n = src.get_number_of_particles()
        x = np.asarray(src.x, dtype=np.float64)
        y = np.asarray(src.y, dtype=np.float64)
        z = np.asarray(src.z, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        return get_particle_array(
            name=name, x=x.copy(), y=y.copy(), z=z.copy(),
            h=np.asarray(src.h, dtype=np.float64).copy(),
            m=np.asarray(src.m, dtype=np.float64).copy(),
            rho=np.ones(n) * self.rho0, p=zeros.copy(),
            cs=np.ones(n) * self.c0,
            u=zeros.copy(), v=zeros.copy(), w=zeros.copy(),
            au=zeros.copy(), av=zeros.copy(), aw=zeros.copy(),
            arho=zeros.copy(), ax=zeros.copy(), ay=zeros.copy(),
            az=zeros.copy(), x0=zeros.copy(), y0=zeros.copy(),
            z0=zeros.copy(), u0=zeros.copy(), v0=zeros.copy(),
            w0=zeros.copy(), rho0=zeros.copy(), backend='warp',
        )

    def create_particles(self):
        self.geom = self._build_geometry()
        cpu_particles = self.geom.create_particles()
        names = ('fluid', 'wall', 'obstacle')
        return tuple(
            self._to_warp(pa, names[i])
            for i, pa in enumerate(cpu_particles)
        )

    def run(self):
        particles = list(self.create_particles())
        fluid = particles[0]
        solids = particles[1:]
        nnps = UniformGridWarpNNPS(
            dim=3, particles=particles, radius_scale=self.radius_scale
        )

        time = 0.0
        for step in range(self.steps):
            scale = damp_factor(step, self.n_damp)
            dt_used = wc_sph_dam_break_step(
                nnps, fluid_index=0,
                solid_indices=tuple(range(1, len(particles))), dt=self.dt,
                rho0=self.rho0, c0=self.c0, p0=self.p0, alpha=self.alpha,
                beta=self.beta, gamma=self.gamma, kernel=self.kernel,
                xsph_eps=self.xsph_eps, gx=0.0, gy=0.0, gz=self.gz,
                gravity_ramp=1.0, adaptive_dt=self.adaptive_dt, cfl=self.cfl,
                dt_min=self.dt_min, dt_max=self.dt_max,
                adaptive_dt_scale=scale, step_dt_max=self.dt_max,
                push=(step == 0), return_dt=True,
            )
            self.dt_history.append(dt_used)
            time += dt_used

        pull = ['x', 'y', 'z', 'rho', 'p', 'cs', 'u', 'v', 'w',
                'au', 'av', 'aw', 'arho']
        fluid.gpu.pull(*pull)
        for solid in solids:
            solid.gpu.pull('x', 'y', 'z', 'rho', 'p')
        metrics = self._metrics(fluid, solids, time)
        if self.output is not None:
            self._write_output(fluid, solids, metrics)
        return metrics

    def _metrics(self, fluid, solids, time):
        wall = solids[0]
        obstacle = solids[1] if len(solids) > 1 else None
        finite_fluid = all(
            np.all(np.isfinite(getattr(fluid, n)))
            for n in ('x', 'y', 'z', 'rho', 'p', 'u', 'v', 'w', 'au', 'av',
                      'aw', 'arho')
        )
        finite_solids = all(
            np.all(np.isfinite(getattr(solid, n)))
            for solid in solids for n in ('x', 'y', 'z', 'rho', 'p')
        )
        ke = 0.5 * float(np.sum(
            fluid.m * (fluid.u**2 + fluid.v**2 + fluid.w**2)
        ))
        dt_hist = np.asarray(self.dt_history)
        return {
            'fluid_particles': int(fluid.get_number_of_particles()),
            'wall_particles': int(wall.get_number_of_particles()),
            'obstacle_particles': (
                0 if obstacle is None else
                int(obstacle.get_number_of_particles())
            ),
            'steps': self.steps,
            'time': float(time),
            'dx': self.dx,
            'hdx': self.hdx,
            'h0': self.h0,
            'rho0': self.rho0,
            'c0': self.c0,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'gz': self.gz,
            'kernel': self.kernel,
            'radius_scale': self.radius_scale,
            'xsph_eps': self.xsph_eps,
            'n_damp': self.n_damp,
            'adaptive_dt': self.adaptive_dt,
            'cfl': self.cfl,
            'dt': self.dt,
            'dt_min_used': float(np.min(dt_hist)),
            'dt_max_used': float(np.max(dt_hist)),
            'dt_last': float(dt_hist[-1]),
            'rho_min': float(np.min(fluid.rho)),
            'rho_max': float(np.max(fluid.rho)),
            'p_min': float(np.min(fluid.p)),
            'p_max': float(np.max(fluid.p)),
            'wall_p_min': float(np.min(wall.p)),
            'wall_p_max': float(np.max(wall.p)),
            'obstacle_p_min': (
                None if obstacle is None else float(np.min(obstacle.p))
            ),
            'obstacle_p_max': (
                None if obstacle is None else float(np.max(obstacle.p))
            ),
            'x_min': float(np.min(fluid.x)),
            'surge_front_x': float(np.max(fluid.x)),
            'z_min': float(np.min(fluid.z)),
            'max_height': float(np.max(fluid.z)),
            'w_mean': float(np.mean(fluid.w)),
            'kinetic_energy': ke,
            'all_finite': bool(finite_fluid and finite_solids),
        }

    def _write_output(self, fluid, solids, metrics):
        wall = solids[0]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            fluid_x=fluid.x, fluid_y=fluid.y, fluid_z=fluid.z,
            fluid_h=fluid.h, fluid_m=fluid.m, fluid_rho=fluid.rho,
            fluid_p=fluid.p, fluid_cs=fluid.cs,
            fluid_u=fluid.u, fluid_v=fluid.v, fluid_w=fluid.w,
            fluid_au=fluid.au, fluid_av=fluid.av, fluid_aw=fluid.aw,
            fluid_arho=fluid.arho,
            wall_x=wall.x, wall_y=wall.y, wall_z=wall.z,
            wall_rho=wall.rho, wall_p=wall.p,
            dt_history=np.asarray(self.dt_history),
            metrics=json.dumps(metrics, sort_keys=True),
        )
        if len(solids) > 1:
            obstacle = solids[1]
            data.update(
                obstacle_x=obstacle.x, obstacle_y=obstacle.y,
                obstacle_z=obstacle.z, obstacle_rho=obstacle.rho,
                obstacle_p=obstacle.p,
            )
        np.savez(self.output, **data)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=H / 15.0,
                   help='Particle spacing (reference uses H/30).')
    p.add_argument('--hdx', type=float, default=1.3)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--rho0', type=float, default=1000.0)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--p0', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=7.0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--kernel', default='wendland',
                   choices=('cubic', 'gaussian', 'wendland'))
    p.add_argument('--radius-scale', type=float, default=2.0)
    p.add_argument('--xsph-eps', type=float, default=0.5)
    p.add_argument('--no-xsph', action='store_true')
    p.add_argument('--gz', type=float, default=-GRAVITY)
    p.add_argument('--n-damp', type=int, default=50)
    p.add_argument('--nboundary-layers', type=int, default=1)
    p.add_argument('--no-adaptive-dt', action='store_true')
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--dt', type=float, default=None)
    p.add_argument('--dt-min', type=float, default=0.0)
    p.add_argument('--dt-max', type=float, default=None)
    p.add_argument('--with-obstacle', action='store_true',
                   help='Include the fixed Kleefsman obstacle as a third array.')
    p.add_argument('--output', default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    runner = WarpDamBreak3DRunner(
        dx=args.dx, hdx=args.hdx, steps=args.steps, rho0=args.rho0,
        c0=args.c0, p0=args.p0, gamma=args.gamma, alpha=args.alpha,
        beta=args.beta, kernel=args.kernel, radius_scale=args.radius_scale,
        xsph_eps=None if args.no_xsph else args.xsph_eps, gz=args.gz,
        n_damp=args.n_damp, nboundary_layers=args.nboundary_layers,
        adaptive_dt=not args.no_adaptive_dt, cfl=args.cfl, dt=args.dt,
        dt_min=args.dt_min, dt_max=args.dt_max,
        with_obstacle=args.with_obstacle, output=args.output,
    )
    metrics = runner.run()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics['fluid_particles'] <= 0:
        raise SystemExit("No fluid particles were created")
    if metrics['wall_particles'] <= 0:
        raise SystemExit("No wall particles were created")
    if args.with_obstacle and metrics['obstacle_particles'] <= 0:
        raise SystemExit("No obstacle particles were created")
    if not metrics['all_finite']:
        raise SystemExit("Non-finite values in final state")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
