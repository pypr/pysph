#!/usr/bin/env python3
"""Warp 3D dam-break surge coupled to a floating rigid box (ADR-0006 P3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DAM_PACKET = HERE.parent / '2026-06-18_warp-dam-break-3d-runner'
sys.path.insert(0, str(DAM_PACKET))
from dam_break_3d_runner import (  # noqa: E402
    GRAVITY, REF_C0, WarpDamBreak3DRunner, damp_factor,
)

from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.warp_nnps import UniformGridWarpNNPS  # noqa: E402
from pysph.base.warp_sph import (  # noqa: E402
    create_rigid_body_state, wc_sph_dam_break_rigid_step,
)


def make_box(dx, h, rho0, body_density=500.0,
             center=(2.35, 0.0, 0.30), size=(0.32, 0.28, 0.20)):
    """Create a shell-sampled rectangular body with physical total mass."""
    axes = []
    for length in size:
        n = max(3, int(round(length / dx)) + 1)
        axes.append(np.linspace(-0.5 * length, 0.5 * length, n))
    ix, iy, iz = np.meshgrid(
        np.arange(len(axes[0])), np.arange(len(axes[1])),
        np.arange(len(axes[2])), indexing='ij')
    shell = ((ix == 0) | (ix == len(axes[0]) - 1) |
             (iy == 0) | (iy == len(axes[1]) - 1) |
             (iz == 0) | (iz == len(axes[2]) - 1))
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    xyz = np.column_stack((X[shell], Y[shell], Z[shell]))
    xyz += np.asarray(center)
    n = len(xyz)
    total_mass = body_density * np.prod(size)
    zeros = np.zeros(n)
    pa = get_particle_array(
        name='body', x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        h=np.full(n, h), m=np.full(n, total_mass / n),
        rho=np.full(n, rho0), p=zeros.copy(), cs=zeros.copy(),
        u=zeros.copy(), v=zeros.copy(), w=zeros.copy(),
        arho=zeros.copy(), fx=zeros.copy(), fy=zeros.copy(), fz=zeros.copy(),
        V=zeros.copy(), rho0=zeros.copy(), backend='warp')
    pa.add_property('body_id', type='int', data=np.zeros(n, dtype=np.int32))
    return pa


def run(args):
    base = WarpDamBreak3DRunner(
        dx=args.dx, hdx=args.hdx, rho0=args.rho0, c0=args.c0,
        alpha=args.alpha, beta=0.0, xsph_eps=args.xsph_eps,
        n_damp=args.n_damp, adaptive_dt=True, cfl=args.cfl,
        nboundary_layers=args.nboundary_layers)
    fluid, wall = base.create_particles()
    body = make_box(
        args.dx, base.h0, args.rho0, body_density=args.body_density,
        center=(args.box_x, 0.0, args.box_z))
    state = create_rigid_body_state(body, nbody=1)
    nnps = UniformGridWarpNNPS(
        dim=3, particles=[fluid, wall, body], radius_scale=2.0)

    initial_xyz = np.column_stack((body.x.copy(), body.y.copy(), body.z.copy()))
    initial_cm = np.average(initial_xyz, axis=0, weights=body.m)
    time_now = 0.0
    dt_history = []
    step = 0
    while step < args.steps and (args.tf is None or time_now < args.tf):
        scale = damp_factor(step, args.n_damp)
        dt_used = wc_sph_dam_break_rigid_step(
            nnps, state, fluid_index=0, wall_indices=(1,), rigid_index=2,
            dt=base.dt, rho0=args.rho0, c0=args.c0, alpha=args.alpha,
            beta=0.0, gamma=7.0, kernel='wendland',
            xsph_eps=args.xsph_eps, gz=-GRAVITY, adaptive_dt=True,
            cfl=args.cfl, dt_min=0.0, dt_max=np.inf,
            adaptive_dt_scale=scale, step_dt_max=np.inf,
            push=(step == 0), return_dt=True)
        dt_history.append(dt_used)
        time_now += dt_used
        step += 1

    fluid.gpu.pull('x', 'y', 'z', 'rho', 'u', 'v', 'w')
    wall.gpu.pull('x', 'y', 'z')
    body.gpu.pull('x', 'y', 'z', 'rho', 'u', 'v', 'w', 'fx', 'fy', 'fz')
    body_xyz = np.column_stack((body.x, body.y, body.z))
    body_vel = np.column_stack((body.u, body.v, body.w))
    final_cm = np.average(body_xyz, axis=0, weights=body.m)
    vc = state.vc.numpy().reshape(1, 3)[0]
    omega = state.omega.numpy().reshape(1, 3)[0]
    d0 = np.linalg.norm(initial_xyz - initial_xyz[0], axis=1)
    d1 = np.linalg.norm(body_xyz - body_xyz[0], axis=1)
    all_finite = all(np.isfinite(a).all() for a in (
        fluid.x, fluid.y, fluid.z, fluid.rho, body_xyz, body_vel,
        body.rho, vc, omega))
    metrics = {
        'fluid_particles': int(len(fluid.x)),
        'wall_particles': int(len(wall.x)),
        'body_particles': int(len(body.x)),
        'steps': step, 'time': float(time_now), 'dx': args.dx,
        'all_finite': bool(all_finite),
        'device_error': int(state.error.numpy()[0]),
        'initial_cm': initial_cm.tolist(), 'final_cm': final_cm.tolist(),
        'cm_displacement': (final_cm - initial_cm).tolist(),
        'vc': vc.tolist(), 'omega': omega.tolist(),
        'body_force': np.sum(np.c_[body.fx, body.fy, body.fz], axis=0).tolist(),
        'body_rho_min': float(body.rho.min()),
        'body_rho_max': float(body.rho.max()),
        'fluid_rho_min': float(fluid.rho.min()),
        'fluid_rho_max': float(fluid.rho.max()),
        'relative_geometry_drift': float(
            np.max(np.abs(d1 - d0)) / max(float(d0.max()), 1e-30)),
        'dt_min': float(np.min(dt_history)),
        'dt_max': float(np.max(dt_history)),
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, metrics=json.dumps(metrics, sort_keys=True),
            fluid_x=fluid.x, fluid_y=fluid.y, fluid_z=fluid.z,
            fluid_u=fluid.u, fluid_v=fluid.v, fluid_w=fluid.w,
            body_x=body.x, body_y=body.y, body_z=body.z,
            body_u=body.u, body_v=body.v, body_w=body.w,
            wall_x=wall.x, wall_y=wall.y, wall_z=wall.z)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=0.10)
    p.add_argument('--hdx', type=float, default=1.3)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--tf', type=float, default=None)
    p.add_argument('--rho0', type=float, default=1000.0)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--xsph-eps', type=float, default=0.5)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--n-damp', type=int, default=50)
    p.add_argument('--nboundary-layers', type=int, default=1)
    p.add_argument('--body-density', type=float, default=500.0)
    p.add_argument('--box-x', type=float, default=2.35)
    p.add_argument('--box-z', type=float, default=0.30)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics['all_finite'] or metrics['device_error']:
        raise SystemExit('coupled run failed finiteness/device-error gate')


if __name__ == '__main__':
    main()
