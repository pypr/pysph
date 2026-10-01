"""Three-dimensional HJC cylinder impacting a frictionless rigid plane.

Run ``pysph run solid_mech.hjc_taylor_bar``. Particle snapshots, history.csv,
damage.png and response.png are written to the output directory.
"""
import json
import os

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.examples.solid_mech.hjc_impact import HJCImpact, concrete_parameters
from pysph.solver.solver import Solver
from pysph.sph.basic_equations import (
    ContinuityEquation, MonaghanArtificialViscosity, VelocityGradient3D)
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate, MomentumEquationWithStress,
    MonaghanArtificialStress)
from pysph.sph.solid_mech.hjc import HJCStep, get_particle_array_hjc


class RigidPlaneContact(Equation):
    """Normal penalty at z=0, acting on particles of radius spacing/2.

    Acceleration is ``stiffness * max(0, spacing/2-z)``. There is no
    adhesion, friction or wall damping. ``stiffness`` has units 1/s^2.
    This analytic plane is local to the example and needs no wall particles.
    """
    def __init__(self, dest, sources, spacing, stiffness):
        self.radius = .5 * spacing
        self.stiffness = stiffness
        super(RigidPlaneContact, self).__init__(dest, sources)

    def loop(self, d_idx, d_z, d_aw, d_wall_acceleration):
        acceleration = self.stiffness * max(0.0, self.radius - d_z[d_idx])
        d_wall_acceleration[d_idx] = acceleration
        d_aw[d_idx] += acceleration


class HJCTaylorBar(HJCImpact):
    def initialize(self):
        self.history = []
        self.impulse = 0.

    def add_user_options(self, group):
        super(HJCTaylorBar, self).add_user_options(group)
        group.add_argument('--wall-factor', type=float, default=1.)

    def consume_user_options(self):
        super(HJCTaylorBar, self).consume_user_options()
        self.wall_factor = self.options.wall_factor
        if not (np.isfinite(self.wall_factor) and self.wall_factor > 0):
            raise ValueError('wall-factor must be finite and positive')
        if self.num_procs != 1:
            raise ValueError('This example records serial contact histories')
        self.n = int(round(.01 / self.dx))
        self.dx = .01 / self.n
        p = concrete_parameters()
        self.cs = np.sqrt((max(p[11]/p[12], p[17]/(1+p[14]))
                           + 4*p[1]/3) / p[0])
        self.stiffness = self.wall_factor * (self.cs / self.dx)**2

    def create_particles(self):
        # Integer subdivisions keep length and initial surface gap fixed.
        n, dx = self.n, self.dx
        xy = (np.arange(n) + .5) * dx - .005
        z_axis = (np.arange(2 * n) + .5) * dx + .0005
        x, y, z = np.meshgrid(xy, xy, z_axis, indexing='ij')
        inside = x*x + y*y < .005**2
        x, y, z = x[inside], y[inside], z[inside]
        kernel = CubicSpline(dim=3)
        pa = get_particle_array_hjc(
            concrete_parameters(), name='concrete', x=x, y=y, z=z,
            w=np.full(x.size, -self.speed), m=np.full(x.size, 2700. * dx**3),
            h=np.full(x.size, 1.3 * dx),
            constants={'wdeltap': kernel.kernel(rij=dx, h=1.3 * dx)})
        pa.add_property('wall_acceleration')
        pa.add_output_arrays(['wall_acceleration'])
        pa.add_constant('spacing', dx)
        self.initial_momentum = -np.sum(pa.m) * self.speed
        self.history = [[0., 0., 0., -self.speed, 0., 0.,
                         self.initial_momentum, 0., .0005]]
        self.case = dict(spacing=dx, particles=x.size, speed=self.speed,
                         radius=.005, length=.02, gap=.0005,
                         wall_factor=self.wall_factor,
                         wall_stiffness=self.stiffness)
        return [pa]

    def create_solver(self):
        cs = self.cs
        dt = .1 * min(1.3 * self.dx / (cs + self.speed),
                      self.dx / (cs * np.sqrt(self.wall_factor)))
        return Solver(
            dim=3, kernel=CubicSpline(dim=3),
            integrator=EPECIntegrator(concrete=HJCStep()),
            dt=dt, tf=6e-5, pfreq=200,
            output_at_times=[0., 2e-5, 4e-5, 6e-5])

    def create_equations(self):
        dest, sources = 'concrete', ['concrete']
        return [
            Group([VelocityGradient3D(dest, sources),
                   MonaghanArtificialStress(dest, None, eps=.3)]),
            Group([ContinuityEquation(dest, sources),
                   MomentumEquationWithStress(dest, sources),
                   MonaghanArtificialViscosity(
                       dest, sources, alpha=1., beta=1.),
                   HookesDeviatoricStressRate(dest, None)]),
            Group([RigidPlaneContact(dest, None, self.dx, self.stiffness)])]

    def post_step(self, solver):
        pa = self.particles[0]
        if (not all(np.isfinite(pa.get(name)).all() for name in
                    ('x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'q', 'damage'))
                or np.min(pa.rho) <= 0 or np.min(pa.damage) < 0
                or np.max(pa.damage) > 1):
            raise RuntimeError('Invalid HJC particle state')
        if (solver.dt * np.max(pa.dt_cfl) / np.min(pa.h) > .2
                or solver.dt * np.sqrt(self.stiffness) > .2):
            raise RuntimeError('Reduce --timestep for acoustic/contact limit')
        gap = np.min(pa.z) - .5 * self.dx
        if gap < -.1 * self.dx:
            raise RuntimeError('Wall penetration exceeds 10% of spacing')
        # EPEC's corrector uses the force evaluated at the half step.
        force = np.sum(pa.m * pa.wall_acceleration)
        self.impulse += solver.dt * force
        momentum = np.sum(pa.m * pa.w)
        residual = momentum - self.initial_momentum - self.impulse
        self.history.append([
            solver.t + solver.dt, solver.t + .5 * solver.dt, force,
            momentum / np.sum(pa.m), np.mean(pa.damage), self.impulse,
            momentum, residual, gap])

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if not self.output_files:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pysph.solver.utils import iter_output

        history_file = os.path.join(self.output_dir, 'history.csv')
        if self.history:
            np.savetxt(history_file, self.history, delimiter=',', comments='',
                       header='time,force_time,force_z,mean_velocity_z,'
                       'mean_damage,wall_impulse,momentum_z,'
                       'momentum_residual,surface_gap')
            with open(os.path.join(self.output_dir, 'case.json'), 'w') as f:
                json.dump(self.case, f, indent=2)
        data = np.genfromtxt(history_file, delimiter=',', names=True)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')
        columns = [('force_z', 1e-3, 'Contact force / kN'),
                   ('mean_velocity_z', 1., 'Axial velocity / m s$^{-1}$'),
                   ('mean_damage', 1., 'Mean damage')]
        for ax, (name, scale, label) in zip(axes, columns):
            time = data['force_time'] if name == 'force_z' else data['time']
            ax.plot(time * 1e6, data[name] * scale)
            ax.set(xlabel=r'Time / $\mu s$', ylabel=label)
            ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(os.path.join(self.output_dir, 'response.png'), dpi=200)
        plt.close(fig)

        frames = []
        for sd, pa in iter_output(self.output_files, 'concrete'):
            frames.append((sd['t'], pa.x.copy(), pa.y.copy(), pa.z.copy(),
                           pa.damage.copy(), pa.q.copy(), pa.spacing[0]))
        snapshots = [min(frames, key=lambda f: abs(f[0] - time))
                     for time in (2e-5, 4e-5, 6e-5)]
        qmax = max(50., np.ceil(max(f[5].max() for f in snapshots) / 5e7)*50)
        fig = plt.figure(figsize=(10, 7), layout='constrained')
        axes = np.array([[fig.add_subplot(2, 3, 3*i+j+1, projection='3d')
                          for j in range(3)] for i in range(2)])
        for j, (time, x, y, z, damage, q, spacing) in enumerate(snapshots):
            keep = y >= 0
            for i, (values, upper) in enumerate(((damage, 1.), (q/1e6, qmax))):
                ax = axes[i, j]
                artist = ax.scatter(
                    x[keep]*1e3, y[keep]*1e3, z[keep]*1e3, c=values[keep],
                    s=5*(spacing/.0005)**2, cmap='viridis', vmin=0, vmax=upper,
                    linewidths=0, depthshade=False, rasterized=True)
                wx, wy = np.meshgrid([-7., 7.], [0., 6.])
                ax.plot_surface(wx, wy, np.zeros_like(wx), color='.7',
                                alpha=.6, shade=False)
                ax.set(xlim=(-7, 7), ylim=(0, 7), zlim=(0, 22))
                ax.set_box_aspect((14, 7, 22))
                ax.set_proj_type('ortho')
                ax.view_init(elev=15, azim=-65)
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(r'$%g\,\mu s$' % (time*1e6))
                if j == 2:
                    fig.colorbar(artist, ax=axes[i, :], shrink=.65,
                                 ticks=[0., upper/2, upper],
                                 label='Damage' if i == 0 else 'q / MPa')
        fig.savefig(os.path.join(self.output_dir, 'damage.png'), dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    app = HJCTaylorBar()
    app.run()
    app.post_process(app.info_filename)
