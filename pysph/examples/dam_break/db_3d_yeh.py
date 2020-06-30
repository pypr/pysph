"""Dam break flow against a tall structure

The case is based of an experiment done by Yeh and Petroff,
University of Washington.
Refer to "Peter E. Raad, Razvan Bidoae, The three-dimensional
Eulerian-Lagrangian marker and micro cell method
for the simulation of free surface flows, Journal of Computational Physics,
Volume 203, Issue 2, 2005, Pages 668-699,
ISSN 0021-9991"

DOI: https://doi.org/10.1016/j.jcp.2004.09.013

"""

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.tools.geometry import remove_overlap_particles
from pysph.base.kernels import CubicSpline
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme

dim = 3

tf = 1.4

H = 0.3
dx = H/25
nboundary_layers = 2
hdx = 1.32
ro = 1000.0
h0 = dx * hdx
gamma = 7.0
alpha = 0.02
beta = 0.0

c0 = 16.0 * np.sqrt(9.81 * H)


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
        self.h0 = self.hdx * self.dx
        self.geom = DamBreak3DGeometry(
            container_height=0.4, container_width=0.61, container_length=1.6,
            fluid_column_height=0.3, fluid_column_width=0.61,
            fluid_column_length=0.4, obstacle_center_x=0.96,
            obstacle_center_y=0, obstacle_length=0.12,
            obstacle_height=0.75, obstacle_width=0.12,
            nboundary_layers=nboundary_layers, with_obstacle=True,
            dx=dx, hdx=hdx, rho0=ro
        )
        self.co = c0

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary', 'obstacle'], dim=dim, rho0=ro, c0=c0,
            h0=h0, hdx=hdx, gz=-9.81, alpha=alpha, beta=beta, gamma=gamma,
            hg_correction=True, delta_sph=True, delta=0.1
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        hdx = self.hdx
        kernel = CubicSpline(dim=dim)
        h0 = self.dx * hdx
        s.configure(h0=h0, hdx=hdx)
        dt = 0.125 * h0 / c0
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=tf, dt=dt,
            adaptive_timestep=True
        )

    def create_particles(self):
        dx = self.dx
        [fluid, boundary, obstacle] = self.geom.create_particles()

        # Thin sheet of water of height 1 cm
        x_sheet, y_sheet, z_sheet = np.mgrid[0.4:1.6:dx, -0.31:0.31:dx, dx:dx+0.01:dx]
        x_sheet = x_sheet.ravel()
        y_sheet = y_sheet.ravel()
        z_sheet = z_sheet.ravel()
        sheet = get_particle_array(x=x_sheet, y=y_sheet, z=z_sheet,
                                   h=self.h0, rho=ro, m=ro*dx*dx*dx)

        remove_overlap_particles(sheet, obstacle, dx_solid=dx)
        remove_overlap_particles(sheet, boundary, dx_solid=dx)
        remove_overlap_particles(sheet, fluid, dx_solid=dx)
        fluid.append_parray(sheet)

        particles = [fluid, boundary, obstacle]
        self.scheme.setup_properties(particles)
        return particles

    def customize_output(self):
        self._mayavi_config('''
        viewer.scalar = 'u'
        b = particle_arrays['boundary']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.1
        ''')

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt
        from pysph.examples import db_exp_data as dbd
        from pysph.tools.interpolator import Interpolator
        import os

        exp_vt, exp_v, exp_ft, exp_f = dbd.get_yeh_petroff_data()
        files = self.output_files

        t = []
        u = []

        # Velocity probe location
        vp_x = 0.814
        vp_y = 0.0
        vp_z = 0.026

        for sd, arrays in iter_output(files, "fluid"):
            t.append(sd["t"])
            interp = Interpolator([arrays],
                                  x=vp_x, y=vp_y, z=vp_z, method="shepard")
            u.append(interp.interpolate('u'))

        u = np.array(u)
        fname = os.path.join(self.output_dir, 'results.npz')
        t, u = list(map(np.asarray, (t, u)))
        np.savez(fname, t=t, u=u)

        t = np.array(t)-0.238
        figure_1 = plt.figure()
        plt.plot(t, u, label="Computed", figure=figure_1)
        plt.scatter(exp_vt, exp_v, label="Experiment, Yeh and Petroff",
                    marker="^", figure=figure_1, color=(0, 0, 0))
        plt.legend()
        plt.ylabel("Horizontal Velocity (m/s)")
        plt.xlabel("Time (s)")
        left, right = plt.xlim()
        plt.xlim(left, 1.4)
        plt.savefig(os.path.join(self.output_dir, 'v_vs_t.png'))
        plt.show()


if __name__ == '__main__':
    app = DamBreak3D()
    app.run()
    app.post_process(app.info_filename)
