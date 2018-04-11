"""
SPHysics case1 - dambreak (6 minutes)
"""

from pysph.base.kernels import (CubicSpline, Gaussian, WendlandQuintic,
                                QuinticSpline)
from pysph.solver.application import Application

from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.base.utils import get_particle_array_wcsph

import numpy as np
from pysph.sph.equation import Group
from pysph.sph.scheme import WCSPHScheme
from pysph.tools.geometry import remove_overlap_particles, rotate
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.wc.kernel_correction import (GradientCorrectionPreStep,
                                            GradientCorrection,
                                            MixedKernelCorrectionPreStep)


def get_dam_geometry(dx_tank=0.03, dx_fluid=0.03, r_tank=100.0, h_f=2.0,
                     l_f=1.0, r_fluid=100.0, hdx=1.5, l_tank=4.0,
                     h_tank=4.0):
    tank_x, tank_y = get_2d_tank(dx_tank, length=l_tank, height=h_tank,
                                 num_layers=2)
    rho_tank = np.ones_like(tank_x) * r_tank
    m_tank = rho_tank * dx_tank * dx_tank
    h_t = np.ones_like(tank_x) * dx_tank * hdx
    tank = get_particle_array_wcsph(name='dam', x=tank_x, y=tank_y, h=h_t,
                                    rho=rho_tank, m=m_tank)
    center = np.array([(l_f - l_tank) / 2.0, h_f / 2.0])
    fluid_x, fluid_y = get_2d_block(dx_fluid, l_f, h_f, center)
    fluid_x += dx_tank
    fluid_y += dx_tank
    h_flu = np.ones_like(fluid_x) * dx_fluid * hdx
    r_f = np.ones_like(fluid_x) * r_fluid
    m_f = r_f * dx_fluid * dx_fluid
    fluid = get_particle_array_wcsph(name='fluid', x=fluid_x, y=fluid_y,
                                     h=h_flu, rho=r_f, m=m_f)
    return fluid, tank


class Dambreak2D(Application):

    def create_particles(self):
        fluid, dam = get_dam_geometry(
            0.05, 0.05, hdx=1.5, h_f=2.0, r_fluid=100.0, r_tank=100.0)
        leng = len(fluid.x) * 9
        fluid.add_property('cwij')
        dam.add_property('cwij')
        fluid.add_constant('m_mat', [0.0] * leng)
        dam.add_constant('m_mat', [0.0] * leng)
        particles = [fluid, dam]
        return particles

    def create_scheme(self):
        co = 10.0 * np.sqrt(2.0 * 9.81 * 2.0)
        return WCSPHScheme(['fluid'], ['dam'], dim=2, rho0=100.0, c0=co, h0=0.075,
                           hdx=1.5, hg_correction=True, gy=-9.81, alpha=0.2,
                           gamma=7.0)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqn1 = Group(equations=[
            GradientCorrectionPreStep('fluid', ['fluid', 'dam'])
        ], real=False)
        for i in range(len(eqns)):
            eqn2 = GradientCorrection('fluid', ['fluid', 'dam'])
            eqns[i].equations.insert(0, eqn2)
        eqns.insert(0, eqn1)
        return eqns

    def configure_scheme(self):
        co = 10.0 * np.sqrt(2.0 * 9.81 * 2.0)
        dt = 0.3 * 0.075 / co
        s = self.scheme
        s.configure_solver(kernel=CubicSpline(dim=2), dt=dt, tf=5.0,
                           adaptive_timestep=False)


if __name__ == '__main__':
    app = Dambreak2D()
    app.run()
