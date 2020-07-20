"""Dam break flow against a tall structure

The case is from "Hakan Akyildiz, N. Erdem Unal,
Sloshing in a three-dimensional rectangular tank:
Numerical simulation and experimental validation,
Ocean Engineering, Volume 33, Issue 16,
2006, Pages 2135-2149, ISSN 0029-8018"
DOI: https://doi.org/10.1016/j.oceaneng.2005.11.001.

pitch angle = 4 deg, roll frequency = 2 rad/s, 75% filled

"""
from math import sin, pi, cos

import os
import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep, IntegratorStep
from pysph.sph.integrator import PECIntegrator

from pysph.sph.equation import Group, Equation

from pysph.sph.scheme import WCSPHScheme
from pysph.examples._db_geometry import DamBreak3DGeometry

Umax = np.sqrt(9.81*0.75*0.62)
c0 = 10.0 * Umax

dx = 0.02
hdx = 1.2

h0 = hdx * dx

tf = 10.0
rho = 1000.0

alpha = 0.1
beta = 0.0
gamma = 7.0

length = 0.92
width = 0.46
height = 0.62
n_layers = 3

theta_0 = 4 * pi/180
omega_r = 2


class PitchingMotion(Equation):
    def __init__(self, dest, sources, theta_0, omega_r):
        self.theta_0 = theta_0
        self.omega_r = omega_r
        super(PitchingMotion, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_aw, t, d_z, d_x):
        theta_0 = self.theta_0
        omega_r = self.omega_r

        omega = theta_0*omega_r*cos(omega_r*t)
        alpha = -theta_0*omega_r*omega_r*sin(omega_r*t)

        at_x = d_z[d_idx]*alpha
        at_z = -d_x[d_idx]*alpha

        ac_x = -d_x[d_idx]*omega*omega
        ac_z = -d_z[d_idx]*omega*omega

        d_au[d_idx] = at_x + ac_x
        d_aw[d_idx] = at_z + ac_z


class OneStageRigidBodyStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
                   d_u, d_v, d_w, d_u0, d_v0, d_w0, d_rho, d_rho0):

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_u, d_v, d_w, d_u0, d_v0, d_w0, d_au, d_av, d_aw,
               dt, d_rho0, d_arho, d_rho):

        d_rho[d_idx] = d_rho0[d_idx] + dt * 0.5 * d_arho[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_u, d_v, d_w, d_u0, d_v0, d_w0, d_au, d_av, d_aw,
               dt, d_rho0, d_arho, d_rho):

        # update velocities
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        # update positions using time-centered velocity
        d_x[d_idx] += dt * 0.5 * (d_u[d_idx] + d_u0[d_idx])
        d_y[d_idx] += dt * 0.5 * (d_v[d_idx] + d_v0[d_idx])
        d_z[d_idx] += dt * 0.5 * (d_w[d_idx] + d_w0[d_idx])

        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]


class SloshingTankPitch(Application):
    def add_user_options(self, group):
        interp_methods = ['shepard', 'sph', 'order1']
        group.add_argument(
            '--dx', action='store', type=float, dest='dx', default=dx,
            help='Particle spacing.'
        )
        group.add_argument(
            '--hdx', action='store', type=float, dest='hdx', default=hdx,
            help='Specify the hdx factor where h = hdx * dx.'
        )
        group.add_argument(
            '--interp-method', action="store", type=str, dest='interp_method',
            default='shepard', help="Specify the interpolation method.",
            choices=interp_methods
        )

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.h0 = self.hdx * self.dx
        self.interp_method = self.options.interp_method

    def create_particles(self):
        geom = DamBreak3DGeometry(
            container_height=height, container_width=width,
            container_length=length, fluid_column_height=height*0.75,
            fluid_column_width=width, fluid_column_length=length,
            nboundary_layers=n_layers, with_obstacle=False,
            dx=self.dx, hdx=self.hdx, rho0=rho
        )

        [fluid, boundary] = geom.create_particles()
        fluid.x = fluid.x - length*0.5
        boundary.x = boundary.x - length*0.5

        # Setting up intital velocity of the tank
        omega0 = theta_0 * omega_r
        boundary.u, boundary.w = boundary.z*omega0, -boundary.x*omega0

        self.scheme.setup_properties([fluid, boundary])
        particles = [fluid, boundary]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=3)
        integrator = PECIntegrator(fluid=WCSPHStep(),
                                   boundary=OneStageRigidBodyStep())

        dt = 0.125 * self.h0 / c0
        self.scheme.configure(h0=self.h0, hdx=self.hdx)

        solver = Solver(kernel=kernel, dim=3, integrator=integrator,
                        tf=tf, dt=dt, adaptive_timestep=True,
                        fixed_h=False)

        return solver

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=3, rho0=rho, c0=c0,
            h0=h0, hdx=hdx, gz=-9.81, alpha=alpha,
            beta=beta, gamma=gamma, hg_correction=True,
            tensile_correction=False, delta_sph=True
        )
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        equation_1 = Group(
            equations=[
                PitchingMotion(
                    dest='boundary', sources=None,
                    theta_0=theta_0, omega_r=omega_r),
            ], real=False
        )

        eqns.insert(0, equation_1)

        return eqns

    def create_tools(self):
        tools = []
        from pysph.solver.tools import DensityCorrection
        rho_corr = DensityCorrection(self, ['fluid', 'boundary'],
                                     corr="shepard", freq=10,
                                     kernel=CubicSpline)

        tools.append(rho_corr)
        return tools

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt
        from pysph.examples import st_exp_data as st
        from pysph.tools.interpolator import Interpolator

        files = self.output_files

        t = []
        p0 = []
        xc0 = 0.0
        for sd, arrays1, arrays2 in iter_output(files, "fluid", "boundary"):
            t.append(sd["t"])
            xc = arrays2.x.mean()
            if sd["t"] == 0:
                xc0 = xc
            # Point 1: Bottom right corner of the tank
            # Point 2: Top right corner of the tank
            # Point 3: Bottom left corner of the tank
            if xc > xc0:
                z1 = arrays2.z.min()
                x2 = arrays2.x.max()
                x1 = arrays2.x[np.where(arrays2.z == z1)[0][0]]
                z2 = arrays2.x[np.where(arrays2.x == x2)[0][0]]
                angle = np.arctan((z2-z1)/(x2-x1))
                x3 = arrays2.x.min()
                z3 = arrays2.z[np.where(arrays2.x == x3)[0][0]]

            if xc <= xc0:
                z2 = arrays2.z.max()
                x1 = arrays2.x.max()
                x2 = arrays2.x[np.where(arrays2.z == z2)[0][0]]
                z1 = arrays2.z[np.where(arrays2.x == x1)[0][0]]
                angle = np.arctan((z2-z1)/(x2-x1)) + pi
                z3 = arrays2.z.min()
                x3 = arrays2.x[np.where(arrays2.z == z3)[0][0]]

            if x3-x1 == 0:
                vec = np.array([-1, 0])
            else:
                vec = np.array([x3-x1, z3-z1])
                vec = vec/np.linalg.norm(vec)

            dx = self.dx
            # Probes at 0.17 m above Point 1 and 0.45 m above Point 3
            x_probe = [x1 + 0.17*cos(angle) + 3.25*dx*vec[0], x3 + 0.45*cos(angle) - 3.25*dx*vec[0]]
            z_probe = [z1 + 0.17*sin(angle) + 3.25*dx*vec[1], z3 + 0.45*sin(angle) - 3.25*dx*vec[1]]
            y_probe = [0, 0]
            interp = Interpolator([arrays1, arrays2], x=x_probe,
                                  y=y_probe, z=z_probe,
                                  method=self.interp_method)
            p0.append(interp.interpolate('p'))

        p0 = np.array(p0)
        p2 = p0.T[0]/1000
        p8 = p0.T[1]/1000

        p2_avg, p8_avg = p2, p8

        fname = os.path.join(self.output_dir, 'results.npz')
        t, p0 = list(map(np.asarray, (t, p0)))
        np.savez(fname, t=t, p0=p0)

        exp_t2, exp_p2, exp_t8, exp_p8 = st.get_au_pitch_data()

        figure_1 = plt.figure()
        # Probe 2
        plt.plot(t[:-10], p2_avg[:-10], label="Computed", figure=figure_1)
        plt.scatter(exp_t2, exp_p2, color=(0, 0, 0),
                    label="Experiment (Akyildiz and Unal, 2006)",
                    figure=figure_1)
        plt.title("P2 v/s t")
        plt.legend()
        plt.ylabel("Pressue [KPa]")
        plt.xlabel("Time [s]")
        plt.xlim(3, 10)
        plt.savefig(os.path.join(self.output_dir, 'p2_vs_t.png'))
        plt.show()

        figure_2 = plt.figure()
        # Probe 8
        plt.plot(t, p8_avg, label="Computed", figure=figure_2)
        plt.scatter(exp_t8, exp_p8, color=(0, 0, 0),
                    label="Experiment (Akyildiz and Unal, 2006)",
                    figure=figure_2)
        plt.title("P8 v/s t")
        plt.legend()
        plt.ylabel("Pressue [KPa]")
        plt.xlabel("Time [s]")
        plt.xlim(3, 10)
        plt.savefig(os.path.join(self.output_dir, 'p8_vs_t.png'))
        plt.show()


if __name__ == '__main__':
    app = SloshingTankPitch()
    app.run()
    app.post_process(app.info_filename)
