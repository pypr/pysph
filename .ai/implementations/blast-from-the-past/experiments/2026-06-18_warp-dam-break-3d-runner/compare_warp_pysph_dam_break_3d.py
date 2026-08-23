#!/usr/bin/env python3
"""Tier-1 smoke parity: Warp 3D dam-break vs a hand-rolled CPU baseline.

Builds the same Lobovsky no-obstacle initial condition (DamBreak3DGeometry,
fluid + one wall array) at a coarse resolution and advances BOTH:

- the Warp ``wc_sph_dam_break_step`` (via ``WarpDamBreak3DRunner``), and
- a hand-rolled CPU EPEC reimplementation using ``LinkedListNNPS(dim=3)`` +
  ``WendlandQuintic(dim=3)`` and the exact same equation blocks the Warp
  generator emits (continuity, Tait + Tait-HG walls, pressure gradient,
  Monaghan AV, XSPH, gravity),

then compares field-by-field. To make the comparison a clean fp32-vs-fp64
diff, both sides run with a FIXED dt, no ``n_damp`` damping, and full gravity
(``adaptive_dt=False``, ``n_damp=0``). The CPU stepper mirrors
``wc_sph_dam_break_step`` exactly: fluid accel + density rate sum over
[fluid, wall]; wall density rate from fluid only; XSPH from fluid only; walls
held fixed (zero acceleration -> the shared PEC stage leaves them in place)
with ``TaitEOSHGCorrection`` (clamp rho >= rho0) re-applied every evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cyarray.carray import UIntArray

from pysph.base.kernels import WendlandQuintic
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.examples._db_geometry import DamBreak3DGeometry

from dam_break_3d_runner import H, REF_C0, WarpDamBreak3DRunner


# --------------------------------------------------------------------------
# CPU equation blocks (mirror the Warp generated blocks; cross-array capable).
# --------------------------------------------------------------------------
def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return nbrs.get_npy_array()[:nbrs.length]


def _xij_vij(dst, src, d, s):
    xij = [dst.x[d] - src.x[s], dst.y[d] - src.y[s], dst.z[d] - src.z[s]]
    vij = [dst.u[d] - src.u[s], dst.v[d] - src.v[s], dst.w[d] - src.w[s]]
    return xij, vij


def _cpu_continuity(nnps, particles, src_index, dst_index, kernel):
    src, dst = particles[src_index], particles[dst_index]
    out = np.zeros(dst.get_number_of_particles())
    for d in range(dst.get_number_of_particles()):
        total = 0.0
        for s in _neighbors(nnps, src_index, dst_index, d):
            xij, vij = _xij_vij(dst, src, d, s)
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d] + src.h[s])
            dw = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dw)
            total += src.m[s] * (vij[0]*dw[0] + vij[1]*dw[1] + vij[2]*dw[2])
        out[d] = total
    return out


def _cpu_pressure_gradient(nnps, particles, src_index, dst_index, kernel):
    src, dst = particles[src_index], particles[dst_index]
    out = np.zeros((dst.get_number_of_particles(), 3))
    for d in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        rhoi21 = 1.0 / (dst.rho[d] * dst.rho[d])
        tmpi = dst.p[d] * rhoi21
        for s in _neighbors(nnps, src_index, dst_index, d):
            xij, _ = _xij_vij(dst, src, d, s)
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d] + src.h[s])
            dw = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dw)
            rhoj21 = 1.0 / (src.rho[s] * src.rho[s])
            acc += -src.m[s] * (tmpi + src.p[s] * rhoj21) * np.asarray(dw)
        out[d] = acc
    return out


def _cpu_artificial_viscosity(nnps, particles, src_index, dst_index, alpha,
                              beta, kernel):
    src, dst = particles[src_index], particles[dst_index]
    out = np.zeros((dst.get_number_of_particles(), 3))
    for d in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        for s in _neighbors(nnps, src_index, dst_index, d):
            xij, vij = _xij_vij(dst, src, d, s)
            vdotx = vij[0]*xij[0] + vij[1]*xij[1] + vij[2]*xij[2]
            if vdotx < 0.0:
                rij2 = xij[0]**2 + xij[1]**2 + xij[2]**2
                rij = np.sqrt(rij2)
                hij = 0.5 * (dst.h[d] + src.h[s])
                mu = hij * vdotx / (rij2 + 0.01*hij*hij)
                rhoij1 = 2.0 / (dst.rho[d] + src.rho[s])
                cij = 0.5 * (dst.cs[d] + src.cs[s])
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                dw = [0.0, 0.0, 0.0]
                kernel.gradient(xij=xij, rij=rij, h=hij, grad=dw)
                acc += -src.m[s] * piij * np.asarray(dw)
        out[d] = acc
    return out


def _cpu_xsph(nnps, particles, src_index, dst_index, eps, kernel):
    src, dst = particles[src_index], particles[dst_index]
    out = np.zeros((dst.get_number_of_particles(), 3))
    if eps is None or eps == 0.0:
        return out
    for d in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        for s in _neighbors(nnps, src_index, dst_index, d):
            xij, vij = _xij_vij(dst, src, d, s)
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d] + src.h[s])
            wij = kernel.kernel(xij=xij, rij=rij, h=hij)
            rhoij1 = 2.0 / (dst.rho[d] + src.rho[s])
            acc += (-eps * src.m[s] * wij * rhoij1) * np.asarray(vij)
        out[d] = acc
    return out


def _tait_eos(rho, rho0, c0, gamma, p0=0.0):
    ratio = rho / rho0
    p = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1.0)
    cs = c0 * ratio**(0.5 * (gamma - 1.0))
    return p, cs


# --------------------------------------------------------------------------
# CPU dam-break EPEC stepper (mirrors wc_sph_dam_break_step exactly).
# --------------------------------------------------------------------------
class CpuDamBreak3D:
    def __init__(self, fluid, wall, rho0, c0, gamma, alpha, beta, xsph_eps,
                 gz, radius_scale, p0=0.0):
        self.fluid = fluid
        self.wall = wall
        self.particles = [fluid, wall]
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = xsph_eps
        self.gz = gz
        self.p0 = p0
        self.radius_scale = radius_scale
        self.kernel = WendlandQuintic(dim=3)

    def _nnps(self):
        return LinkedListNNPS(
            dim=3, particles=self.particles, radius_scale=self.radius_scale
        )

    def _accel(self):
        f, w = self.fluid, self.wall
        # EOS: fluid Tait; wall Tait-HG (clamp rho >= rho0 in place).
        f.p[:], f.cs[:] = _tait_eos(
            f.rho, self.rho0, self.c0, self.gamma, self.p0
        )
        np.maximum(w.rho, self.rho0, out=w.rho)
        w.p[:], w.cs[:] = _tait_eos(w.rho, self.rho0, self.c0, self.gamma)
        # Zero accumulators (walls keep zero accel -> stay fixed).
        for pa in (f, w):
            for name in ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'):
                getattr(pa, name)[:] = 0.0
        nnps = self._nnps()
        # Fluid: continuity + pressure + AV summed over [fluid, wall].
        for src in (0, 1):
            f.arho[:] += _cpu_continuity(nnps, self.particles, src, 0,
                                         self.kernel)
            pg = _cpu_pressure_gradient(nnps, self.particles, src, 0,
                                        self.kernel)
            av = _cpu_artificial_viscosity(nnps, self.particles, src, 0,
                                           self.alpha, self.beta, self.kernel)
            f.au[:] += pg[:, 0] + av[:, 0]
            f.av[:] += pg[:, 1] + av[:, 1]
            f.aw[:] += pg[:, 2] + av[:, 2]
        # XSPH from fluid neighbours only.
        xs = _cpu_xsph(nnps, self.particles, 0, 0, self.eps, self.kernel)
        f.ax[:], f.ay[:], f.az[:] = xs[:, 0], xs[:, 1], xs[:, 2]
        # Wall density rate from the fluid only.
        w.arho[:] = _cpu_continuity(nnps, self.particles, 0, 1, self.kernel)
        # Gravity (full) into the fluid acceleration.
        f.aw[:] += self.gz

    def _save_state(self):
        for pa in (self.fluid, self.wall):
            pa.x0[:], pa.y0[:], pa.z0[:] = pa.x, pa.y, pa.z
            pa.u0[:], pa.v0[:], pa.w0[:] = pa.u, pa.v, pa.w
            pa.rho0[:] = pa.rho

    def _pec_stage(self, dt, stage, xsph):
        fac = dt * stage
        for pa in (self.fluid, self.wall):
            use_xsph = xsph and (pa is self.fluid)
            ax = pa.ax if use_xsph else np.zeros_like(pa.x)
            ay = pa.ay if use_xsph else np.zeros_like(pa.x)
            az = pa.az if use_xsph else np.zeros_like(pa.x)
            # Position uses the CURRENT velocity (before this stage's kick).
            new_x = pa.x0 + fac * (pa.u + ax)
            new_y = pa.y0 + fac * (pa.v + ay)
            new_z = pa.z0 + fac * (pa.w + az)
            pa.u[:] = pa.u0 + fac * pa.au
            pa.v[:] = pa.v0 + fac * pa.av
            pa.w[:] = pa.w0 + fac * pa.aw
            pa.x[:], pa.y[:], pa.z[:] = new_x, new_y, new_z
            pa.rho[:] = pa.rho0 + fac * pa.arho

    def step(self, dt):
        self._save_state()
        self._accel()                         # E
        self._pec_stage(dt, 0.5, xsph=True)   # P
        self._accel()                         # E
        self._pec_stage(dt, 1.0, xsph=True)   # C

    def run(self, steps, dt):
        for _ in range(steps):
            self.step(dt)


def _build_cpu_particles(dx, hdx, rho0, c0, nboundary_layers):
    geom = DamBreak3DGeometry(
        container_height=1.5 * H, container_width=H / 2.0,
        container_length=161 * H / 30.0, fluid_column_height=H,
        fluid_column_width=H / 2.0, fluid_column_length=2.0 * H,
        dx=dx, nboundary_layers=nboundary_layers, hdx=hdx, rho0=rho0,
        with_obstacle=False,
    )
    fluid_cpu, wall_cpu = geom.create_particles()

    def rebuild(src, name):
        n = src.get_number_of_particles()
        z = np.zeros(n)
        return get_particle_array(
            name=name,
            x=np.asarray(src.x).copy(), y=np.asarray(src.y).copy(),
            z=np.asarray(src.z).copy(), h=np.asarray(src.h).copy(),
            m=np.asarray(src.m).copy(), rho=np.ones(n) * rho0,
            p=z.copy(), cs=np.ones(n) * c0,
            u=z.copy(), v=z.copy(), w=z.copy(),
            au=z.copy(), av=z.copy(), aw=z.copy(), arho=z.copy(),
            ax=z.copy(), ay=z.copy(), az=z.copy(),
            x0=z.copy(), y0=z.copy(), z0=z.copy(),
            u0=z.copy(), v0=z.copy(), w0=z.copy(), rho0=z.copy(),
        )

    return rebuild(fluid_cpu, 'fluid'), rebuild(wall_cpu, 'wall')


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dx', type=float, default=0.12)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--dt', type=float, default=1.0e-4)
    p.add_argument('--hdx', type=float, default=1.3)
    p.add_argument('--rho0', type=float, default=1000.0)
    p.add_argument('--c0', type=float, default=REF_C0)
    p.add_argument('--gamma', type=float, default=7.0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--xsph-eps', type=float, default=0.5)
    p.add_argument('--gz', type=float, default=-9.81)
    p.add_argument('--radius-scale', type=float, default=2.0)
    p.add_argument('--nboundary-layers', type=int, default=1)
    p.add_argument('--rtol', type=float, default=2.0e-3,
                   help='Relative tol for kinematics + density.')
    p.add_argument('--p-atol-factor', type=float, default=16.0,
                   help='Pressure absolute tol = factor*rho0*c0^2*2^-23 '
                        '(fp32 Tait-EOS cancellation floor near rest).')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--prefix', default='comparison-smoke')
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    warp_path = out_dir / f'{args.prefix}-warp.npz'

    # --- Warp side: fixed dt, no damping, full gravity (clean diff) ---
    runner = WarpDamBreak3DRunner(
        dx=args.dx, hdx=args.hdx, steps=args.steps, rho0=args.rho0,
        c0=args.c0, gamma=args.gamma, alpha=args.alpha, beta=args.beta,
        kernel='wendland', radius_scale=args.radius_scale,
        xsph_eps=args.xsph_eps, gz=args.gz, n_damp=0, adaptive_dt=False,
        dt=args.dt, nboundary_layers=args.nboundary_layers, output=warp_path,
    )
    warp_metrics = runner.run()
    warp = np.load(warp_path)

    # --- CPU side: identical IC (deterministic geometry), same fixed dt ---
    fluid, wall = _build_cpu_particles(
        args.dx, args.hdx, args.rho0, args.c0, args.nboundary_layers
    )
    cpu = CpuDamBreak3D(
        fluid, wall, rho0=args.rho0, c0=args.c0, gamma=args.gamma,
        alpha=args.alpha, beta=args.beta, xsph_eps=args.xsph_eps, gz=args.gz,
        radius_scale=args.radius_scale,
    )
    cpu.run(args.steps, args.dt)

    # Sanity: identical particle counts / initial layout.
    assert fluid.get_number_of_particles() == warp['fluid_x'].size, (
        "fluid count mismatch -> non-identical IC"
    )
    assert wall.get_number_of_particles() == warp['wall_x'].size, (
        "wall count mismatch -> non-identical IC"
    )

    diffs = {
        'fluid_x': _max_abs_diff(fluid.x, warp['fluid_x']),
        'fluid_y': _max_abs_diff(fluid.y, warp['fluid_y']),
        'fluid_z': _max_abs_diff(fluid.z, warp['fluid_z']),
        'fluid_u': _max_abs_diff(fluid.u, warp['fluid_u']),
        'fluid_v': _max_abs_diff(fluid.v, warp['fluid_v']),
        'fluid_w': _max_abs_diff(fluid.w, warp['fluid_w']),
        'fluid_rho': _max_abs_diff(fluid.rho, warp['fluid_rho']),
        'fluid_p': _max_abs_diff(fluid.p, warp['fluid_p']),
        'wall_rho': _max_abs_diff(wall.rho, warp['wall_rho']),
        'wall_p': _max_abs_diff(wall.p, warp['wall_p']),
    }
    # Relative scales for a fp32-vs-fp64 pass/fail verdict.
    scales = {
        'fluid_x': np.abs(warp['fluid_x']).max() + 1.0,
        'fluid_y': np.abs(warp['fluid_y']).max() + 1.0,
        'fluid_z': np.abs(warp['fluid_z']).max() + 1.0,
        'fluid_u': np.abs(warp['fluid_u']).max() + 1.0,
        'fluid_v': np.abs(warp['fluid_v']).max() + 1.0,
        'fluid_w': np.abs(warp['fluid_w']).max() + 1.0,
        'fluid_rho': float(args.rho0),
        'fluid_p': np.abs(warp['fluid_p']).max() + 1.0,
        'wall_rho': float(args.rho0),
        'wall_p': np.abs(warp['wall_p']).max() + 1.0,
    }
    rel = {k: float(diffs[k] / scales[k]) for k in diffs}
    worst = float(max(rel.values()))
    cpu_finite = all(np.all(np.isfinite(getattr(fluid, n)))
                     for n in ('x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p'))

    # Verdict. Kinematics + density are compared in RELATIVE terms -- the Warp
    # (fp32) vs CPU (fp64) step agrees on these to ~1e-7. Pressure is special
    # near rest: p = B*((rho/rho0)^gamma - 1) is a tiny difference of near-equal
    # quantities, so the fp32 rounding of rho (~rho0*2^-23 absolute) is
    # amplified by dp/drho ~ B*gamma/rho0, giving a large RELATIVE p error while
    # the ABSOLUTE error stays at the fp32 Tait cancellation floor. We therefore
    # gate pressure on an absolute tolerance derived from that floor, not a
    # relative one. (At a developed free surface, p >> this floor and the
    # relative agreement recovers; see tier-2 / experiment.md.)
    kin_fields = ('fluid_x', 'fluid_y', 'fluid_z', 'fluid_u', 'fluid_v',
                  'fluid_w', 'fluid_rho', 'wall_rho')
    fp32_eps = 2.0 ** -23
    p_abs_floor = args.p_atol_factor * args.rho0 * args.c0 * args.c0 * fp32_eps
    kin_ok = all(rel[k] <= args.rtol for k in kin_fields)
    pressure_ok = (diffs['fluid_p'] <= p_abs_floor
                   and diffs['wall_p'] <= p_abs_floor)
    passed = bool(
        bool(warp_metrics['all_finite']) and bool(cpu_finite)
        and kin_ok and pressure_ok
    )

    report = {
        'params': {
            'dx': args.dx, 'steps': args.steps, 'dt': args.dt,
            'hdx': args.hdx, 'rho0': args.rho0, 'c0': args.c0,
            'gamma': args.gamma, 'alpha': args.alpha, 'beta': args.beta,
            'xsph_eps': args.xsph_eps, 'gz': args.gz,
            'radius_scale': args.radius_scale,
        },
        'fluid_particles': int(fluid.get_number_of_particles()),
        'wall_particles': int(wall.get_number_of_particles()),
        'max_abs_diff': diffs,
        'max_rel_diff': rel,
        'worst_rel_diff': worst,
        'rtol_kinematic': args.rtol,
        'pressure_abs_floor': float(p_abs_floor),
        'kinematics_density_ok': bool(kin_ok),
        'pressure_ok': bool(pressure_ok),
        'warp_metrics': warp_metrics,
        'passed': passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / f'{args.prefix}-summary.json').write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    if not passed:
        raise SystemExit("Tier-1 parity failed (see max_rel_diff)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
