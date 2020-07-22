"""Shallow water sloshing

The case is as described in "Bouscasse, B., Antuono, M.,
Colagrossi, A., & Lugni, C. (2013). Numerical and Experimental
Investigation of Nonlinear Shallow Water Sloshing,
International Journal of Nonlinear Sciences and Numerical Simulation,
14(2), 123-138"

DOI:10.1515/ijnsns-2012-0100

This case is same as the Series 5 (Large amplitude sway motion with
a small water depth) case mentioned in the above paper.
"""

from math import cos, pi, exp, sqrt, tanh, sin

import os
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Equation
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.wc.edac import EDACScheme, EDACStep


L = 1  # Length of tank
h = 0.03 * L  # Height of water
amp = 2.333 * h  # Amplitude of oscillation

u_max = 1.3
c0 = 10.0 * u_max

dx = h/6
hdx = 1.2

h0 = hdx * dx
n_layers = 4

tf = 33
rho = 1000.0

k = pi/L
omega_r = sqrt(9.81*k*tanh(k*h))
omega = omega_r*1.231


class HorizontalExcitation(Equation):
    def __init__(self, dest, sources):
        L = 1
        h = 0.03 * L
        self.amp = 2.333 * h

        k = pi/L
        omega_r = sqrt(9.81*k*tanh(k*h))
        self.omega = omega_r*1.231
        super(HorizontalExcitation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, t, d_rho):
        amp = self.amp
        omega = self.omega

        d_au[d_idx] += amp * (omega) * (omega) * sin(omega*t)


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
        h0 = self.h0
        volume = dx * dx
        m = rho * volume

        xt, yt = get_2d_tank(
            dx=dx, length=L, height=0.2*L,
            num_layers=n_layers, base_center=[L/2, -dx]
            )

        xf, yf = get_2d_block(
            dx=dx, length=L - 2*dx, height=h, center=[L/2, h/2]
            )

        fluid = get_particle_array(name='fluid', x=xf, y=yf, h=h0,
                                   m=m, rho=rho)
        solid = get_particle_array(name='solid', x=xt, y=yt, h=h0,
                                   m=m, rho=rho)

        fluid.u = -amp * omega * np.ones_like(xf)

        self.scheme.setup_properties([fluid, solid])
        particles = [fluid, solid]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = EPECIntegrator(fluid=EDACStep())

        dt = 0.125 * self.h0 / c0
        self.scheme.configure(h=self.h0)
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=tf, dt=dt)

        return solver

    def create_scheme(self):
        s = EDACScheme(['fluid'], ['solid'], dim=2, rho0=rho, c0=c0, gy=-9.81,
                       alpha=0.0, nu=0.0, h=h0, clamp_p=True)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqns[1].equations.insert(
            -1, HorizontalExcitation(dest='fluid', sources=None)
            )
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
            probe_y = np.linspace(0, arrays1.y.max(), 50)
            probe = get_particle_array(
                x=probe_x*np.ones_like(probe_y), y=probe_y
                )
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
        wave_height = wave_height - h

        import pysph.examples.st_exp_data as st

        exp_t, exp_H = st.get_bouscasse_data()
        T = 2*pi/omega
        t = np.array(t)/T
        plt.plot(t, wave_height, label="Computed")
        plt.scatter(exp_t, exp_H, label="Experiment: Bouscasse et al, 2013")
        plt.xlabel("t/T")
        plt.ylabel("Wave Height (m)")
        plt.xlim(8.5, 11)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'H_vs_t.png'))
        plt.show()


if __name__ == '__main__':
    app = SloshingTank()
    app.run()
    app.post_process(app.info_filename)
