"""Three-dimensional dam break over a dry bed. (14 hours)

The case is described as a SPHERIC benchmark
 https://wiki.manchester.ac.uk/spheric/index.php/Test2

By default the simulation runs for 6 seconds of simulation time.
"""

import numpy as np

from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme

dim = 3

dt = 1e-5
tf = 6.0

# parameter to change the resolution
dx = 0.02
nboundary_layers = 1
hdx = 1.3
ro = 1000.0
h0 = dx * hdx
gamma = 7.0
alpha = 0.25
beta = 0.0
c0 = 10.0 * np.sqrt(2.0 * 9.81 * 0.55)


class DamBreak3D(Application):
    def add_user_options(self, group):
        group.add_argument(
            '--dx', action='store', type=float, dest='dx',  default=dx,
            help='Particle spacing.'
        )
        group.add_argument(
            '--hdx', action='store', type=float, dest='hdx',  default=hdx,
            help='Specify the hdx factor where h = hdx * dx.'
        )

    def consume_user_options(self):
        dx = self.options.dx
        self.dx = dx
        self.hdx = self.options.hdx
        self.geom = DamBreak3DGeometry(
            dx=dx, nboundary_layers=nboundary_layers, hdx=self.hdx, rho0=ro
        )
        self.co = 10.0 * self.geom.get_max_speed(g=9.81)

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary', 'obstacle'], dim=dim, rho0=ro, c0=c0,
            h0=h0, hdx=hdx, gz=-9.81, alpha=alpha, beta=beta, gamma=gamma,
            hg_correction=True, tensile_correction=False
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        hdx = self.hdx
        kernel = WendlandQuintic(dim=dim)
        h0 = self.dx * hdx
        s.configure(h0=h0, hdx=hdx)
        dt = 0.25*h0/(1.1 * self.co)
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=tf, dt=dt,
            adaptive_timestep=True, n_damp=50,
            output_at_times=[0.4, 0.6, 1.0]
        )

    def create_particles(self):
        return self.geom.create_particles()

    def customize_output(self):
        self._mayavi_config('''
        viewer.scalar = 'u'
        b = particle_arrays['boundary']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.1
        ''')


if __name__ == '__main__':
    app = DamBreak3D()
    app.run()
