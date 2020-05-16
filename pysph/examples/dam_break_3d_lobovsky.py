"""Three-dimensional dam break. (14 hours)

The case is as described in "L.Lobovsky, E.Botia-Vera,
 F.Castellana, J.Mas-Soler, A.Souto-Iglesias,
Experimental investigation of dynamic pressure loads during dam break,
J. Fluids Struct. 48 (2014) 407-434."

DOI:10.1016/j.jfluidstructs.2014.03.009
"""

import numpy as np

from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme

dim = 3

dt = 1e-5
tf = 2.5

# parameter to change the resolution

H = 1.0
dx = H/30.0
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
                                       container_height=1.5*H,
                                       container_width=H/2.0,
                                       container_length=161*H/30,
                                       fluid_column_height=H,
                                       fluid_column_width=H/2.0,
                                       fluid_column_length=2.0*H,
                                       dx=dx,
                                       nboundary_layers=nboundary_layers,
                                       hdx=self.hdx, rho0=ro,
                                       with_obstacle=False
                                      )
        self.co = 10.0 * self.geom.get_max_speed(g=9.81)

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=dim, rho0=ro, c0=c0,
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

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt
        from pysph.examples import db_exp_data as dbd
        from pysph.tools.interpolator import Interpolator
        import os
        H = self.geom.fluid_column_height
        factor_y = 1/(ro*9.81*H)
        factor_x = np.sqrt(9.81/H)

        t1, t2, t3, data_p1, data_p2, data_p3 = dbd.get_lobovsky_data()
        files = self.output_files

        t = []
        p0 = []
        p_x = np.repeat(self.geom.container_length, 3)
        p_y = np.repeat(0, 3)
        p_z = np.array([H/100, H/10, 8*H/30])

        from mayavi import mlab

        for sd, arrays1, arrays2 in iter_output(files, "fluid", "boundary"):
            t.append(sd["t"]*factor_x)
            interp = Interpolator([arrays1, arrays2],
                                  x=p_x, y=p_y, z=p_z, method="shepard")
            p0.append(interp.interpolate('p')*factor_y)

        fname = os.path.join(self.output_dir, 'results.npz')
        t, p0 = list(map(np.asarray, (t, p0)))
        np.savez(fname, t=t, p0=p0)

        p1 = p0[:, 0]
        p2 = p0[:, 1]
        p3 = p0[:, 2]

        idx = t <= 7
        t = t[idx]
        p1 = p1[idx]
        p2 = p2[idx]
        p3 = p3[idx]

        fig1 = plt.figure()
        plt.plot(t, p1, label="p1 computed", figure=fig1)
        plt.plot(t1, data_p1, label="Lobovsky et al.", figure=fig1)
        plt.legend()
        plt.ylabel(r"$\frac{P}{\rho gH}$")
        plt.xlabel(r"$t \sqrt{\frac{g}{H}} $")
        plt.title("P1")
        plt.savefig(os.path.join(self.output_dir, 'p1_vs_t.png'))

        fig2 = plt.figure()
        plt.plot(t, p2, label="p2 computed", figure=fig2)
        plt.plot(t2, data_p2, label="Lobovsky et al.", figure=fig2)
        plt.legend()
        plt.ylabel(r"$\frac{P}{\rho gH}$")
        plt.xlabel(r"$t \sqrt{\frac{g}{H}} $")
        plt.title("P2")
        plt.savefig(os.path.join(self.output_dir, 'p2_vs_t.png'))

        fig3 = plt.figure()
        plt.plot(t, p3, label="p3 computed", figure=fig3)
        plt.plot(t3, data_p3, label="Lobovsky et al.", figure=fig3)
        plt.legend()
        plt.ylabel(r"$\frac{P}{\rho gH}$")
        plt.xlabel(r"$t \sqrt{\frac{g}{H}} $")
        plt.title("P3")
        plt.savefig(os.path.join(self.output_dir, 'p3_vs_t.png'))


if __name__ == '__main__':
    app = DamBreak3D()
    app.run()
    app.post_process(app.info_filename)
