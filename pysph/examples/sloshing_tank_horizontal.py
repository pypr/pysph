"""Liquid sloshing under horizontal excitation

The case is as described in "ODD M. FALTINSEN, OLAV F. ROGNEBAKKE, IVAN A.
LUKOVSKY, and ALEXANDER N. TIMOKHA. Multidimensional
modal analysis of nonlinear sloshing in a rectangular tank with finite
water depth. Journal of Fluid Mechanics, 407:201234, 2000."

DOI: https://doi.org/10.1017/S0022112099007569


This case is with h=0.6 m and T=1.3 s
"""

# math
from math import cos, pi, exp

import os
import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import OneStageRigidBodyStep

from pysph.sph.equation import Group, Equation
from pysph.tools.geometry import get_2d_tank, get_2d_block

from pysph.sph.scheme import WCSPHScheme

Umax = 2*np.sqrt(9.81*0.6)
c0 = 10.0 * Umax

dx = 0.01
hdx = 1.3
h0 = hdx * dx

n_layers = 3
tf = 10.0
rho = 1000.0

alpha = 0.1
beta = 0.0
gamma = 7.0

length = 1.73
h_tank = 1.15
h_liquid = 0.6

amp = 0.032
T = 1.3


class HorizontalExcitation(Equation):
    def __init__(self, dest, sources, amp, T):
        self.amp = amp
        self.T = T
        self.pi = pi
        super(HorizontalExcitation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, t):
        amp = self.amp
        T = self.T
        pi = self.pi

        d_au[d_idx] = -amp * (2*pi/T) * (2*pi/T) * cos(2*pi*t/T)


class SloshingTank(Application):
    def add_user_options(self, group):
        group.add_argument(
            '--dx', action='store', type=float, dest='dx', default=dx,
            help='Particle spacing.'
        )
        group.add_argument(
            '--hdx', action='store', type=float, dest='hdx', default=hdx,
            help='Specify the hdx factor where h = hdx * dx.'
        )

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.h0 = self.hdx * self.dx

    def create_particles(self):
        dx = self.dx
        h0 = self.hdx * self.dx
        m = rho * dx * dx

        xt, yt = get_2d_tank(dx=dx, length=length, height=h_tank,
                             num_layers=n_layers, base_center=[0.0, -dx])

        xf, yf = get_2d_block(dx=dx, length=length - 2*dx,
                              height=h_liquid, center=[0.0, h_liquid*0.5])

        fluid = get_particle_array(name='fluid', x=xf, y=yf, h=h0,
                                   m=m, rho=rho)
        solid = get_particle_array(name='solid', x=xt, y=yt, h=h0,
                                   m=m, rho=rho)

        self.scheme.setup_properties([fluid, solid])
        particles = [fluid, solid]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = PECIntegrator(fluid=WCSPHStep(),
                                   solid=OneStageRigidBodyStep())

        dt = 0.5*self.dx/(1.1*c0)
        self.scheme.configure(h0=self.h0, hdx=self.hdx)

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=tf, dt=dt, adaptive_timestep=True,
                        fixed_h=False)

        return solver

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['solid'], dim=2, rho0=rho, c0=c0,
            h0=h0, hdx=hdx, gy=-9.81, alpha=alpha,
            beta=beta, gamma=gamma, hg_correction=True,
            tensile_correction=False
        )
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()
        equation_1 = Group(
            equations=[
                HorizontalExcitation(
                    dest='solid', sources=None, amp=amp, T=T),
            ], real=False
        )

        eqns.insert(0, equation_1)

        return eqns

    def post_process(self, info_fname):
        """
        Comparing wave height 0.05 m to the left of the tank
        """
        self.read_info(info_fname)

        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt
        from cyarray.api import UIntArray
        from pysph.base import nnps

        files = self.output_files
        wave_height, t = [], []
        for sd, arrays1, arrays2 in iter_output(files, 'fluid', 'solid'):
            t.append(sd['t'])
            x_left = arrays2.x.min()
            probe_x = x_left + 0.05 + self.dx * (n_layers - 1)
            probe_y = np.linspace(0.0, arrays1.y.max(), 50)
            probe = get_particle_array(x=probe_x*np.ones_like(probe_y), y=probe_y)
            pa_list = [arrays1, probe]
            nps = nnps.LinkedListNNPS(dim=2, particles=pa_list, radius_scale=1)
            src_index, dst_index = 0, 1
            nps.set_context(src_index=0, dst_index=1)
            nbrs = UIntArray()
            wh = 0.0
            for i in range(len(probe_y)):
                nps.get_nearest_particles(src_index, dst_index, i, nbrs)
                neighbours = nbrs.get_npy_array()
                for j in range(len(neighbours)):
                    if arrays1.y[neighbours[j]] > wh:
                        wh = arrays1.y[neighbours[j]]

            wave_height.append(wh)

        wave_height = np.array(wave_height)
        wave_height = wave_height - h_liquid

        import pysph.examples.st_exp_data as st

        exp_t, exp_H = st.get_faltinsen_data()

        plt.plot(t, wave_height, label="Computed")
        plt.plot(exp_t, exp_H, label="Experiment: Faltinsen et al, 2000")
        plt.xlabel("T (s)")
        plt.ylabel("Wave Height (m)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'H_vs_t.png'))
        plt.show()


if __name__ == '__main__':
    app = SloshingTank()
    app.run()
    app.post_process(app.info_filename)
