"""Two-dimensional dam break over a dry bed.  (30 minutes)

The case is described in "State of the art classical SPH for free surface
flows", Moncho Gomez-Gesteira, Benedict D Rogers, Robert A, Dalrymple and Alex
J.C Crespo, Journal of Hydraulic Research, Vol 48, Extra Issue (2010), pp
6-27. DOI:10.1080/00221686.2010.9641242

"""

import os
import numpy as np

from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak2DGeometry
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme

fluid_column_height = 2.0
fluid_column_width = 1.0
container_height = 4.0
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
        group.add_argument(
            "--h-factor", action="store", type=float, dest="h_factor",
            default=1.0,
            help="Divide default h by this factor to change resolution"
        )

    def consume_user_options(self):
        self.h = h/self.options.h_factor
        print("Using h = %f"%self.h)
        self.hdx = self.hdx
        self.dx = self.h/self.hdx

    def configure_scheme(self):
        self.scheme.configure(h0=self.h, hdx=self.hdx)
        kernel = WendlandQuintic(dim=2)
        tf = 2.5
        from pysph.sph.integrator import EPECIntegrator
        self.scheme.configure_solver(
            kernel=kernel,
            integrator_cls=EPECIntegrator,
            tf=tf,
            adaptive_timestep=True, n_damp=50, fixed_h=False
        )

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=2, rho0=ro, c0=co,
            h0=h, hdx=1.3, gy=-9.81, alpha=alpha, beta=beta,
            gamma=gamma, hg_correction=True, update_h=True
        )
        return s

    def create_particles(self):
        geom = DamBreak2DGeometry(
            container_width=container_width, container_height=container_height,
            fluid_column_height=fluid_column_height,
            fluid_column_width=fluid_column_width, dx=self.dx, dy=self.dx,
            nboundary_layers=1, ro=ro, co=co,
            with_obstacle=False,
            beta=1.0, nfluid_offset=1, hdx=self.hdx)
        return geom.create_particles()

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
