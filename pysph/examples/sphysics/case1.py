"""
SPHysics case1 - dambreak (6 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.base.utils import get_particle_array

import numpy as np
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles, rotate
from pysph.tools.geometry import get_2d_tank, get_2d_block


def get_dam_geometry(dx_tank=0.03, dx_fluid=0.03, r_tank=100.0, h_f=2.0,
                     l_f=1.0, r_fluid=100.0, hdx=1.5, l_tank=4.0,
                     h_tank=4.0):
    tank_x, tank_y = get_2d_tank(dx_tank, length=l_tank, height=h_tank,
                                 num_layers=5)
    rho_tank = np.ones_like(tank_x) * r_tank
    m_tank = rho_tank * dx_tank * dx_tank
    h_t = np.ones_like(tank_x) * dx_tank * hdx
    tank = get_particle_array(name='dam', x=tank_x, y=tank_y, h=h_t,
                              rho=rho_tank, m=m_tank)
    center = np.array([(l_f - l_tank) / 2.0, h_f / 2.0])
    fluid_x, fluid_y = get_2d_block(dx_fluid, l_f, h_f, center)
    fluid_x += dx_tank
    fluid_y += dx_tank
    h_fluid = np.ones_like(fluid_x) * dx_fluid * hdx
    r_f = np.ones_like(fluid_x) * r_fluid
    m_f = r_f * dx_fluid * dx_fluid
    fluid = get_particle_array(name='fluid', x=fluid_x, y=fluid_y, h=h_fluid,
                               rho=r_f, m=m_f)
    return fluid, tank


class Dambreak2D(Application):

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.3,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.03,
            help="spacing between the particles")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.h0 = self.hdx * self.dx
        self.dt = 0.15 * self.h0 / co

    def create_particles(self):
        fluid, dam = get_dam_geometry(self.dx, self.dx, hdx=self.hdx,
                                      h_f=h_fluid, r_fluid=ro, r_tank=ro)
        self.scheme.setup_properties([fluid, dam])
        particles = [fluid, dam]
        return particles

    def create_scheme(self):
        aha = AdamiHuAdamsScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co,
                                 alpha=alp, gy=-9.81, nu=0.0, h0=0.03,
                                 gamma=1.0)
        wcsph = WCSPHScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co,
                            h0=0.03, hdx=1.3, hg_correction=True,
                            gy=-9.81, alpha=alp, gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co, gy=-9.81,
                          alpha=0.0, nu=0.0, h=0.03, clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, aha=aha, edac=edac)

    def configure_scheme(self):
        s = self.scheme
        scheme = self.options.scheme
        if scheme == 'wcsph':
            s.configure(h0=self.h0, hdx=self.hdx)
        elif scheme == 'aha':
            s.configure(h0=self.h0)
        elif scheme == 'edac':
            s.configure(h=self.h0)
        s.configure_solver(kernel=CubicSpline(dim=2), dt=self.dt, tf=3.0,
                           adaptive_timestep=False)


if __name__ == '__main__':
    l_dam = 4.0
    h_dam = 4.0
    h_fluid = 2.0
    l_fluid = 1.0
    gamma = 7.0
    alp = 0.2
    ro = 100.0
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    app = Dambreak2D()
    app.run()
