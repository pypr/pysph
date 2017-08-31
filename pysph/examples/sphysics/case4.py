"""
SPHysics case4 - tsunami (20 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import TransportVelocityStep, WCSPHStep
from pysph.sph.integrator_step import OneStageRigidBodyStep
from pysph.sph.integrator_step import TwoStageRigidBodyStep
from pysph.sph.integrator import Integrator
from pysph.base.utils import get_particle_array

import numpy as np
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.geometry import get_2d_block
from pysph.examples.sphysics.beach_geometry import get_beach_geometry_2d


def get_tsunami_geometry(dx_solid=0.01, dx_fluid=0.01, r_solid=100.0,
                         r_fluid=100.0, hdx=1.3, l_wall=9.5, h_wall=4.0,
                         angle=26.565051, h_fluid=3.5, l_obstacle=0.91,
                         flat_l=2.25):
    x1, y1, x2, y2 = get_beach_geometry_2d(dx_solid, l_wall, h_wall, flat_l,
                                           angle, 4)
    wall_x = np.concatenate([x1, x2])
    wall_y = np.concatenate([y1, y2])
    r_wall = np.ones_like(wall_x) * r_solid
    m_wall = r_wall * dx_solid * dx_solid
    h_w = np.ones_like(wall_x) * dx_solid * hdx
    wall = get_particle_array(name='wall', x=wall_x, y=wall_y, h=h_w,
                              rho=r_wall, m=m_wall)
    theta = np.pi * angle / 180.0
    h_obstacle = l_obstacle * np.tan(theta)
    obstacle_x1, obstacle_y1 = get_2d_block(dx_solid, l_obstacle, h_obstacle)
    obstacle_x = []
    obstacle_y = []
    for x, y in zip(obstacle_x1, obstacle_y1):
        if (y >= (-x * np.tan(theta))):
            obstacle_x.append(x)
            obstacle_y.append(y)
    x_translate = (l_wall - flat_l) * 0.8
    y_translate = x_translate * np.tan(theta)
    obstacle_x = np.asarray(obstacle_x) - x_translate
    obstacle_y = np.asarray(obstacle_y) + y_translate + dx_solid
    h_obstacle = np.ones_like(obstacle_x) * dx_solid * hdx
    r_obstacle = np.ones_like(obstacle_x) * r_solid
    m_obstacle = r_obstacle * dx_solid * dx_solid
    obstacle = get_particle_array(name='obstacle', x=obstacle_x,
                                  y=obstacle_y, rho=r_obstacle,
                                  m=m_obstacle, h=h_obstacle)
    fluid_center = np.array([flat_l - l_wall / 2.0, h_fluid / 2.0])
    x_fluid, y_fluid = get_2d_block(dx_f, l_wall, h_fluid, fluid_center)
    x3 = []
    y3 = []
    for i, xi in enumerate(x_fluid):
        if y_fluid[i] >= np.tan(-theta) * xi:
            x3.append(xi)
            y3.append(y_fluid[i])
    fluid_x = np.array(x3)
    fluid_y = np.array(y3)
    h_f = np.ones_like(fluid_x) * dx_fluid * hdx
    r_f = np.ones_like(fluid_x) * r_fluid
    m_f = r_f * dx_fluid * dx_fluid
    fluid = get_particle_array(name='fluid', x=fluid_x, y=fluid_y,
                               h=h_f, m=m_f, rho=r_f)
    remove_overlap_particles(fluid, obstacle, dx_solid * 1.1, 2)
    remove_overlap_particles(fluid, wall, dx_solid * 1.1, 2)
    return fluid, wall, obstacle


class Tsunami2D(Application):

    def initialize(self):
        self.count = 0

    def pre_step(self, solver):
        if self.count == 0 and solver.t >= 2.5:
            obstacle = self.particles[2]
            obstacle.v[:] = 0.0
            obstacle.u[:] = 0.0
            self.count += 1

    def create_particles(self):
        a = 2.0
        theta = 26.565051 * np.pi / 180.0
        c = np.cos(theta)
        s = np.sin(theta)
        fluid, wall, obstacle = get_tsunami_geometry(dx_s, dx_f, hdx=hdx,
                                                     h_fluid=h_fluid)
        obstacle.u[:] = a * c
        obstacle.v[:] = -a * s
        self.scheme.setup_properties([fluid, wall, obstacle])
        if self.options.scheme == 'aha':
            for p in ['x0', 'y0', 'z0']:
                obstacle.add_property(p)
        if self.options.scheme == 'edac':
            for p in ['rho0', 'arho']:
                fluid.add_property(p)
            fluid.rho0[:] = ro
            fluid.add_output_arrays(['rho0', 'arho'])
        particles = [fluid, wall, obstacle]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = Integrator(
            fluid=WCSPHStep(), obstacle=TwoStageRigidBodyStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=10.0, dt=dt, adaptive_timestep=False,
                        output_at_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                         8.0, 9.0, 10.0])
        return solver

    def create_scheme(self):
        aha = AdamiHuAdamsScheme(['fluid'], ['wall', 'obstacle'], dim=2,
                                 rho0=ro, c0=co, alpha=alp, gy=-9.81, nu=0.0,
                                 h0=h0, gamma=1.0)
        wcsph = WCSPHScheme(['fluid'], ['wall', 'obstacle'], dim=2, rho0=ro,
                            c0=co, h0=h0, hdx=hdx, hg_correction=True,
                            gy=-9.81, alpha=alp, gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['wall', 'obstacle'], dim=2, rho0=ro,
                          c0=co, gy=-9.81, alpha=alp, nu=0.0, h=h0,
                          clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, aha=aha, edac=edac)


if __name__ == '__main__':
    h_fluid = 3.0
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    ro = 100.0
    alp = 0.1
    gamma = 7.0
    dx_s = 0.05
    dx_f = 0.05
    hdx = 1.3
    h0 = max(dx_s, dx_f) * hdx
    dt = 0.25 * h0 / co
    print 'dt=%s, h0=%s, hdx=%s, co=%s' % (dt, h0, hdx, co)
    app = Tsunami2D()
    app.run()
