"""Head-on impact of two concrete blocks with HJC damage (plane strain).

Run with ``pysph run solid_mech.hjc_impact``. The output directory contains
particle snapshots, response.csv, damage.png and response.png.
"""
import os

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.basic_equations import (
    ContinuityEquation, MonaghanArtificialViscosity, VelocityGradient2D)
from pysph.sph.equation import Group
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate, MomentumEquationWithStress,
    MonaghanArtificialStress)
from pysph.sph.solid_mech.hjc import (
    HJCStep, get_particle_array_hjc, hjc_parameters)


def concrete_parameters():
    # Illustrative consistent SI parameters; not an experimental calibration.
    return hjc_parameters(
        rho0=2700., G=2.417e10, A=.29, B=2.06, C=.0013, N=.866,
        fc=1.19e8, T=8.2e6, edot0=1., efmin=.01, sfmax=5.,
        pc=4e7, muc=.00124, pl=1.2e9, mul=.011, D1=.04, D2=1.,
        K1=1.287e10, K2=1.631e10, K3=6.495e10)


class HJCImpact(Application):
    def add_user_options(self, group):
        group.add_argument('--spacing', type=float, default=0.0005)
        group.add_argument('--speed', type=float, default=30.)

    def consume_user_options(self):
        self.dx = self.options.spacing
        self.speed = self.options.speed
        if not (np.isfinite(self.dx) and 0 < self.dx <= .001
                and np.isfinite(self.speed) and self.speed > 0):
            raise ValueError('Require 0 < spacing <= 0.001 and speed > 0')

    def create_particles(self):
        # The interface is x=0; opposing velocities initiate compression.
        n = int(round(.01 / self.dx))
        dx = .01 / n
        self.dx = dx
        axis = (np.arange(-n, n) + .5) * dx
        x, y = np.meshgrid(axis, axis)
        x, y = x.ravel(), y.ravel()
        kernel = CubicSpline(dim=2)
        pa = get_particle_array_hjc(
            concrete_parameters(), name='concrete', x=x, y=y,
            u=np.where(x < 0, self.speed, -self.speed),
            rho=np.full(x.size, 2700.), m=np.full(x.size, 2700. * dx**2),
            h=np.full(x.size, 1.3 * dx),
            constants={'wdeltap': kernel.kernel(rij=dx, h=1.3 * dx)})
        pa.add_property('side', data=np.where(x < 0, -1., 1.))
        pa.add_output_arrays(['side'])
        return [pa]

    def create_solver(self):
        p = concrete_parameters()
        cs = np.sqrt((p[11] / p[12] + 4 * p[1] / 3) / p[0])
        return Solver(
            dim=2, kernel=CubicSpline(dim=2),
            integrator=EPECIntegrator(concrete=HJCStep()),
            dt=.05 * 1.3 * self.dx / (cs + self.speed), tf=1.2e-5,
            pfreq=100, output_at_times=[0., 4e-6, 8e-6, 1.2e-5])

    def create_equations(self):
        dest, sources = 'concrete', ['concrete']
        return [
            Group(equations=[VelocityGradient2D(dest, sources),
                             MonaghanArtificialStress(dest, None, eps=.3)]),
            Group(equations=[ContinuityEquation(dest, sources),
                             MomentumEquationWithStress(dest, sources),
                             MonaghanArtificialViscosity(
                                 dest, sources, alpha=1., beta=1.),
                             HookesDeviatoricStressRate(dest, None)])]

    def post_step(self, solver):
        pa = self.particles[0]
        if (not all(np.isfinite(pa.get(name)).all() for name in
                    ('x', 'y', 'u', 'v', 'rho', 'p', 'q', 'damage'))
                or np.min(pa.rho) <= 0 or np.min(pa.damage) < 0
                or np.max(pa.damage) > 1):
            raise RuntimeError('Invalid HJC particle state')
        if solver.dt * np.max(pa.dt_cfl) / np.min(pa.h) > .2:
            raise RuntimeError('Reduce --timestep to keep acoustic CFL <= 0.2')

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if not self.output_files:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pysph.solver.utils import iter_output

        frames, rows = [], []
        for sd, pa in iter_output(self.output_files, 'concrete'):
            t = sd['t']
            left = pa.side < 0
            rows.append([t, np.mean(pa.u[left]), np.mean(pa.damage),
                         np.max(pa.damage), np.sum(pa.m * pa.u),
                         .5 * np.sum(pa.m * (pa.u**2 + pa.v**2))])
            frames.append((t, pa.x.copy(), pa.y.copy(), pa.damage.copy(),
                           pa.q.copy()))
        data = np.asarray(rows)
        np.savetxt(os.path.join(self.output_dir, 'response.csv'), data,
                   delimiter=',', header='time,left_velocity,mean_damage,'
                   'max_damage,momentum_x,kinetic_energy', comments='')
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')
        for j, target in enumerate((4e-6, 8e-6, 1.2e-5)):
            frame = min(frames, key=lambda f: abs(f[0] - target))
            t, x, y, damage, q = frame
            fields = ((damage, 1.), (q / 1e6, 400.))
            for i, (values, vmax) in enumerate(fields):
                ax = axes[i, j]
                artist = ax.scatter(x * 1e3, y * 1e3, c=values, s=7,
                                    cmap='viridis', vmin=0., vmax=vmax,
                                    linewidths=0, rasterized=True)
                ax.set(aspect='equal', xlim=(-11, 11), ylim=(-11, 11))
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(r'$%g\,\mu s$' % (t * 1e6))
                if j == 2:
                    fig.colorbar(artist, ax=axes[i, :], shrink=.8,
                                 ticks=[0., vmax / 2, vmax],
                                 label='Damage' if i == 0 else 'q / MPa')
        fig.savefig(os.path.join(self.output_dir, 'damage.png'), dpi=200)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')
        axes[0].plot(data[:, 0] * 1e6, data[:, 1])
        axes[1].plot(data[:, 0] * 1e6, data[:, 2])
        labels = ('Mean left-block velocity / m s$^{-1}$', 'Mean damage')
        for ax, label in zip(axes, labels):
            ax.set(xlabel=r'Time / $\mu s$', ylabel=label)
            ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(os.path.join(self.output_dir, 'response.png'), dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    app = HJCImpact()
    app.run()
    app.post_process(app.info_filename)
