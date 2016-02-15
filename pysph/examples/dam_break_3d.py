"""Three-dimensional dam break over a dry bed. (14 hours)

The case is described as a SPHERIC benchmark
 https://wiki.manchester.ac.uk/spheric/index.php/Test2

By default the simulation runs for 6 seconds of simulation time.
"""

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
nboundary_layers=3
hdx = 1.2
ro = 1000.0
h0 = dx * hdx
gamma = 7.0
alpha = 0.5
beta = 0.0

class DamBreak3D(Application):
    def initialize(self):
        self.geom = DamBreak3DGeometry(
            dx=dx, nboundary_layers=nboundary_layers, hdx=hdx, rho0=ro
        )
        self.co = 10.0 * self.geom.get_max_speed(g=9.81)

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=dim, rho0=ro, c0=self.co,
            h0=h0, hdx=hdx, gz=-9.81, alpha=alpha, beta=beta, gamma=gamma,
            hg_correction=True, tensile_correction=True
        )
        kernel = WendlandQuintic(dim=dim)
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=tf, dt=dt,
            adaptive_timestep=True, n_damp=50
        )
        return s

    def create_particles(self):
        return self.geom.create_particles()


if __name__ == '__main__':
    app = DamBreak3D()
    app.run()
