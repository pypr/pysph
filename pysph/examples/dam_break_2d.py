"""Two-dimensional dam break over a dry bed.  (30 minutes)

The case is described in "State of the art classical SPH for free surface
flows", Moncho Gomez-Gesteira, Benedict D Rogers, Robert A, Dalrymple and Alex
J.C Crespo, Journal of Hydraulic Research, Vol 48, Extra Issue (2010), pp
6-27. DOI:10.1080/00221686.2010.9641242

"""

import os
import numpy as np

from pysph.examples._db_geometry import DamBreak2DGeometry

from pysph.base.kernels import WendlandQuintic
from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation, \
    UpdateSmoothingLengthFerrari, ContinuityEquationDeltaSPH

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator, EPECIntegrator, TVDRK3Integrator
from pysph.sph.integrator_step import WCSPHStep, WCSPHTVDRK3Step

fluid_column_height = 2.0
fluid_column_width = 1.0
container_height = 3.0
container_width = 4.0
nboundary_layers = 2
#h = 0.0156
h = 0.039
ro = 1000.0
co = 10.0 * np.sqrt(2*9.81*fluid_column_height)
gamma = 7.0
alpha = 0.1
beta = 0.0
B = co*co*ro/gamma
p0 = 1000.0

class DamBreak2D(Application):
    def add_user_options(self, group):
        self.hdx = 1.3
        group.add_option(
            "--h-factor", action="store", type=float, dest="h_factor",
            default=1.0,
            help="Divide default h by this factor to change resolution"
        )

    def consume_user_options(self):
        self.h = h/self.options.h_factor
        print("Using h = %f"%self.h)
        self.hdx = self.hdx
        self.dx = self.h/self.hdx

    def create_particles(self):
        geom = DamBreak2DGeometry(
            container_width=container_width, container_height=container_height,
            fluid_column_height=fluid_column_height,
            fluid_column_width=fluid_column_width, dx=self.dx, dy=self.dx,
            nboundary_layers=1, ro=ro, co=co,
            with_obstacle=False,
            beta=2.0, nfluid_offset=1, hdx=self.hdx)
        return geom.create_particles()

    def create_solver(self):
        dt = 0.125*self.h/co
        print("dt = %f"%dt)
        tf = 2.5

        dim = 2
        # Create the kernel
        kernel = WendlandQuintic(dim=dim)

        # Create the Integrator. Currently, PySPH supports multi-stage,
        # predictor corrector and a TVD-RK3 integrators.

        #integrator = EPECIntegrator(fluid=WCSPHStep(), boundary=WCSPHStep())
        integrator = TVDRK3Integrator(fluid=WCSPHTVDRK3Step(), boundary=WCSPHTVDRK3Step())

        # Create a solver.  The damping is performed for the first 50 iterations.
        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=True, n_damp=50,
                        fixed_h=False)

        return solver

    def create_equations(self):

        # create the equations
        equations = [

            # Equation of state
            Group(equations=[

                    TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma),
                    TaitEOSHGCorrection(dest='boundary', sources=None, rho0=ro, c0=co, gamma=gamma),
                    ], real=False),

            Group(equations=[

                    # Continuity equation with dissipative corrections for fluid on fluid
                    ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=co, delta=0.1),
                    ContinuityEquation(dest='fluid', sources=['boundary']),
                    ContinuityEquation(dest='boundary', sources=['fluid']),

                    # Momentum equation
                    MomentumEquation(dest='fluid', sources=['fluid', 'boundary'],
                                    alpha=alpha, beta=beta, gy=-9.81, c0=co,
                                    tensile_correction=True),

                    # Position step with XSPH
                    XSPHCorrection(dest='fluid', sources=['fluid'])

                    ]),

            # smoothing length update
            Group( equations=[
                    UpdateSmoothingLengthFerrari(dest='fluid', sources=None, hdx=self.hdx, dim=2)
                    ], real=True ),
            ]
        return equations

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        t, x_max = [], []
        factor = np.sqrt(2.0*9.81/fluid_column_width)
        files = self.output_files
        for sd, array in iter_output(files, 'fluid'):
            t.append(sd['t']*factor)
            x = array.get('x')
            x_max.append(x.max())

        t, x_max = list(map(np.asarray, (t, x_max)))
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, x_max=x_max)

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from pysph.examples import db_exp_data as dbd
        plt.plot(t, x_max, label='Computed')
        te, xe = dbd.get_koshizuka_oka_data()
        plt.plot(te, xe, 'o', label='Koshizuka & Oka (1996)')
        plt.xlim(0, 0.7*factor); plt.ylim(0, 4.5)
        plt.xlabel('$T$'); plt.ylabel('$Z/L$')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output_dir, 'x_vs_t.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    app = DamBreak2D()
    app.run()
    app.post_process(app.info_filename)
