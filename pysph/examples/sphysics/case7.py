"""
SPHysics case7 - wavemaker in beach with stationary obstacle (25 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator_step import TwoStageRigidBodyStep
from pysph.base.utils import get_particle_array

import numpy as np
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.geometry import get_2d_block
from pysph.examples.sphysics.beach_geometry import get_beach_geometry_2d


def get_wavespaddle_geometry(hdx=1., dx_f=0.02, dx_s=0.02, r_f=100., r_s=100.,
                             length=13., height=0.8, flat_l=1.48, h_fluid=0.43,
                             angle=2.8624):
    x1, y1, x2, y2 = get_beach_geometry_2d(dx_s, length, height, flat_l,
                                           angle, 3)
    theta = np.pi * angle / 180.0
    p1x = flat_l - 9.605
    p1y = 0.40625
    p2x = flat_l - 10.065
    p2y = 0.6085
    theta1 = np.arctan((p2y - p1y) / (p2x - p1x))
    p3x = flat_l - 10.28
    p3y = 0.6085
    theta2 = np.arctan((p3y - p2y) / (p3x - p2x))
    p4x = flat_l - 10.62
    p4y = 0.457
    theta3 = np.arctan((p4y - p3y) / (p4x - p3x))
    obs_x1 = np.arange(p2x, p1x, dx_s / 2.0)
    obs_y1 = p1y + (obs_x1 - p1x) * np.tan(theta1)
    obs_x2 = np.arange(p3x, p2x, dx_s / 2.0)
    obs_y2 = p2y + (obs_x2 - p2x) * np.tan(-theta2)
    obs_x3 = np.arange(p4x, p3x, dx_s / 2.0)
    obs_y3 = p3y + (obs_x3 - p3x) * np.tan(theta3)
    x1 = np.concatenate([x1, obs_x1, obs_x2, obs_x3])
    y1 = np.concatenate([y1, obs_y1, obs_y2, obs_y3])
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

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.3,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.02,
            help="spacing between the particles")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.h0 = self.hdx * self.dx
        self.dt = 0.25 * self.h0 / co

    def pre_step(self, solver):
        t = solver.t
        theta = 2.0 * np.pi * t / period
        paddle = self.particles[2]
        paddle.u[:] = -amplitude * np.sin(theta)

    def create_particles(self):
        fluid, wall, paddle = get_wavespaddle_geometry(
            self.hdx, self.dx, self.dx, h_fluid=h_fluid)
        self.scheme.setup_properties([fluid, wall, paddle])
        scheme = self.options.scheme
        if scheme == 'aha' or scheme == 'edac':
            for p in ['u0', 'v0', 'w0', 'x0', 'y0', 'z0']:
                paddle.add_property(p)
        particles = [fluid, wall, paddle]
        return particles

    def create_scheme(self):
        aha = AdamiHuAdamsScheme(['fluid'], ['wall', 'paddle'], dim=2,
                                 rho0=ro, c0=co, alpha=alp, gy=-9.81, nu=0.0,
                                 h0=0.02, gamma=1.0)
        wcsph = WCSPHScheme(['fluid'], ['wall', 'paddle'], dim=2, rho0=ro,
                            c0=co, h0=0.02, hdx=1.3, hg_correction=True,
                            gy=-9.81, alpha=alp, gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['wall', 'paddle'], dim=2, rho0=ro,
                          c0=co, gy=-9.81, alpha=alp, nu=0.0, h=0.02,
                          clamp_p=True)
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
        step = dict(paddle=TwoStageRigidBodyStep())
        s.configure_solver(
            kernel=CubicSpline(dim=2), tf=12.0, dt=self.dt,
            adaptive_timestep=False, extra_steppers=step)


if __name__ == '__main__':
    h_fluid = 0.43
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    ro = 100.0
    alp = 0.2
    gamma = 7.0
    flat_l = 1.0
    amplitude = 1.0
    period = 0.6
    app = WavesPaddle2D()
    app.run()
