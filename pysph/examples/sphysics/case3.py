"""
SPHysics case2 - wavemaker in beach (20 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
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


def get_wavespaddle_geometry(hdx=1.5, dx_f=0.1, dx_s=0.1, r_f=100., r_s=100.,
                             length=3.75, height=0.3, flat_l=1., angle=4.2364,
                             h_fluid=0.2):
    x1, y1, x2, y2 = get_beach_geometry_2d(dx_s, length, height, flat_l,
                                           angle, 5)
    r1 = np.ones_like(x1) * r_s
    m1 = r1 * dx_s * dx_s
    h1 = np.ones_like(x1) * hdx * dx_s
    wall = get_particle_array(name='wall', x=x1, y=y1, rho=r1, m=m1, h=h1)
    r2 = np.ones_like(x2) * r_s
    m2 = r2 * dx_s * dx_s
    h2 = np.ones_like(x2) * hdx * dx_s
    paddle = get_particle_array(name='paddle', x=x2, y=y2, rho=r2, m=m2, h=h2)
    fluid_center = np.array([flat_l - length / 2.0, h_fluid / 2.0])
    x_fluid, y_fluid = get_2d_block(dx_f, length, h_fluid, fluid_center)
    x3 = []
    y3 = []
    theta = np.pi * angle / 180.0
    for i, xi in enumerate(x_fluid):
        if y_fluid[i] >= np.tan(-theta) * xi:
            x3.append(xi)
            y3.append(y_fluid[i])
    x3 = np.array(x3)
    y3 = np.array(y3)
    r3 = np.ones_like(x3) * r_f
    m3 = r3 * dx_f * dx_f
    h3 = np.ones_like(x3) * hdx * dx_f
    fluid = get_particle_array(name='fluid', x=x3, y=y3, rho=r3, m=m3, h=h3)
    remove_overlap_particles(fluid, wall, dx_s, 2)
    remove_overlap_particles(fluid, paddle, dx_s, 2)
    return fluid, wall, paddle


class WavesPaddle2D(Application):

    def pre_step(self, solver):
        t = solver.t
        theta = 2.0 * np.pi * t / period
        paddle = self.particles[2]
        paddle.u = amplitude * (paddle.y - dx_s) * np.cos(theta)
        paddle.v = amplitude * (flat_l - paddle.x) * np.cos(theta)

    def create_particles(self):
        fluid, wall, paddle = get_wavespaddle_geometry(hdx, dx_f, dx_s,
                                                       h_fluid=h_fluid)
        self.scheme.setup_properties([fluid, wall, paddle])
        if self.options.scheme == 'aha':
            for p in ['x0', 'y0', 'z0']:
                paddle.add_property(p)
        if self.options.scheme == 'edac':
            for p in ['rho0', 'arho']:
                fluid.add_property(p)
            fluid.rho0[:] = ro
            fluid.add_output_arrays(['rho0', 'arho'])
        particles = [fluid, wall, paddle]
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = Integrator(fluid=WCSPHStep(),
                                paddle=TwoStageRigidBodyStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=10.0, dt=dt, adaptive_timestep=False,
                        output_at_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                         8.0, 9.0, 10.0])
        return solver

    def create_scheme(self):
        aha = AdamiHuAdamsScheme(['fluid'], ['wall', 'paddle'], dim=2,
                                 rho0=ro, c0=co, alpha=alp, gy=-9.81, nu=0.0,
                                 h0=h0, gamma=1.0)
        wcsph = WCSPHScheme(['fluid'], ['wall', 'paddle'], dim=2, rho0=ro,
                            c0=co, h0=h0, hdx=hdx, hg_correction=True,
                            gy=-9.81, alpha=alp, gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['wall', 'paddle'], dim=2, rho0=ro,
                          c0=co, gy=-9.81, alpha=alp, nu=0.0, h=h0,
                          clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, aha=aha, edac=edac)


if __name__ == '__main__':
    h_fluid = 0.25
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    flat_l = 1.0
    dx_s = 0.01
    dx_f = 0.01
    alp = 0.1
    gamma = 7.0
    ro = 100.0
    alp = 0.1
    hdx = 1.3
    h0 = max(dx_s, dx_f) * hdx
    dt = 0.25 * h0 / co
    amplitude = 1.0
    period = 1.4
    print 'dt=%s, h0=%s, hdx=%s, co=%s' % (dt, h0, hdx, co)
    app = WavesPaddle2D()
    app.run()
