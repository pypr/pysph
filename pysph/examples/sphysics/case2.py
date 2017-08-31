"""
SPHysics case2 - dambreak on wet surface (40 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator import Integrator
from pysph.base.utils import get_particle_array

import numpy as np
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles, rotate
from pysph.tools.geometry import get_2d_tank, get_2d_block


def get_dam_geometry(dx_tank=0.03, dx_fluid=0.03, r_tank=100.0, h_f=2.0,
                     l_f=1.0, r_fluid=100.0, hdx=1.5, l_tank=4.0,
                     h_tank=4.0, h_f2=1.0):
    tank_x, tank_y = get_2d_tank(dx_tank, length=l_tank, height=h_tank,
                                 num_layers=4)
    rho_tank = np.ones_like(tank_x) * r_tank
    m_tank = rho_tank * dx_tank * dx_tank
    h_t = np.ones_like(tank_x) * dx_tank * hdx
    tank = get_particle_array(name='dam', x=tank_x, y=tank_y, h=h_t,
                              rho=rho_tank, m=m_tank)
    center = np.array([(l_f - l_tank) / 2.0, h_f / 2.0])
    fluid_x1, fluid_y1 = get_2d_block(dx_fluid, l_f, h_f, center)
    center = np.array([l_f / 2.0, h_f2 / 2.0])
    fluid_x2, fluid_y2 = get_2d_block(dx_fluid, l_tank - l_f - 2.0 * dx_fluid,
                                      h_f2, center)
    fluid_x = np.concatenate([fluid_x1, fluid_x2])
    fluid_y = np.concatenate([fluid_y1, fluid_y2])
    h_fluid = np.ones_like(fluid_x) * dx_fluid * hdx
    r_f = np.ones_like(fluid_x) * r_fluid
    m_f = r_f * dx_fluid * dx_fluid
    fluid = get_particle_array(name='fluid', x=fluid_x, y=fluid_y,
                               h=h_fluid, rho=r_f, m=m_f)
    remove_overlap_particles(fluid, tank, dx_tank, 2)
    return fluid, tank


class Dambreak_2D(Application):

    def create_particles(self):
        fluid, dam = get_dam_geometry(dx_s, dx_f, hdx=hdx, h_f=h_fluid,
                                      h_f2=h_fluid2, r_fluid=ro, r_tank=ro,
                                      l_f=l_fluid, l_tank=l_dam, h_tank=h_dam)
        self.scheme.setup_properties([fluid, dam])
        if self.options.scheme == 'edac':
            for p in ['rho0', 'arho']:
                fluid.add_property(p)
            fluid.rho0[:] = ro
            fluid.add_output_arrays(['rho0', 'arho'])
        particles = [fluid, dam]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = Integrator(fluid=WCSPHStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=10.0, dt=dt, adaptive_timestep=False,
                        output_at_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                         8.0, 9.0, 10.0])
        return solver

    def create_scheme(self):
        aha = AdamiHuAdamsScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co,
                                 alpha=alp, gy=-9.81, nu=0.0, h0=h0,
                                 gamma=1.0)
        wcsph = WCSPHScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co, h0=h0,
                            hdx=hdx, hg_correction=True, gy=-9.81, alpha=alp,
                            gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['dam'], dim=2, rho0=ro, c0=co, gy=-9.81,
                          alpha=alp, nu=0.0, h=h0, clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, aha=aha, edac=edac)


if __name__ == '__main__':
    l_dam = 2.0
    h_dam = 0.16
    h_fluid = 0.15
    l_fluid = 0.376
    h_fluid2 = 0.018
    gamma = 7.0
    alp = 0.1
    ro = 100.0
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    dx_s = 0.005
    dx_f = 0.005
    hdx = 1.3
    h0 = max(dx_s, dx_f) * hdx
    dt = 0.15 * h0 / co
    app = Dambreak_2D()
    app.run()
