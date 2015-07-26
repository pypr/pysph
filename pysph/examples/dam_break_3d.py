"""Three-dimensional dam break over a dry bed. (14 hours)

The case is described as a SPHERIC benchmark
 https://wiki.manchester.ac.uk/spheric/index.php/Test2

By default the simulation runs for 6 seconds of simulation time.
"""

from pysph.examples._db_geometry import DamBreak3DGeometry

from pysph.base.kernels import WendlandQuintic

from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

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
        self.co = co = 10.0 * self.geom.get_max_speed(g=9.81)
        self.B = co*co*ro/gamma

    def create_particles(self):
        # the geometry generator
        return self.geom.create_particles()

    def create_solver(self):

        # Create the kernel
        kernel = WendlandQuintic(dim=dim)

        # Setup the integrator.
        integrator = EPECIntegrator(
            fluid=WCSPHStep(), boundary=WCSPHStep(), obstacle=WCSPHStep()
        )

        # Create a solver.
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator, tf=tf, dt=dt,
            adaptive_timestep=True, n_damp=50
        )
        return solver

    def create_equations(self):
        # create the equations
        co = self.co
        equations = [

            # Equation of state
            Group(equations=[

                    TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma),
                    TaitEOSHGCorrection(dest='boundary', sources=None, rho0=ro, c0=co, gamma=gamma),
                    TaitEOSHGCorrection(dest='obstacle', sources=None, rho0=ro, c0=co, gamma=gamma),

                    ], real=False),

            # Continuity, momentum and xsph equations
            Group(equations=[

                    ContinuityEquation(dest='fluid', sources=['fluid', 'boundary', 'obstacle']),
                    ContinuityEquation(dest='boundary', sources=['fluid']),
                    ContinuityEquation(dest='obstacle', sources=['fluid']),

                    MomentumEquation(dest='fluid', sources=['fluid', 'boundary', 'obstacle'],
                                    alpha=alpha, beta=beta, gz=-9.81, c0=co,
                                    tensile_correction=True),

                    XSPHCorrection(dest='fluid', sources=['fluid'])

                    ]),
            ]
        return equations

if __name__ == '__main__':
    app = DamBreak3D()
    app.run()
