"""
SPHysics case6 - wavemaker in beach with moving obstacles (30 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator_step import TwoStageRigidBodyStep
from pysph.base.utils import get_particle_array, get_particle_array_rigid_body

import numpy as np
from pysph.sph.equation import Group
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.geometry import get_2d_block
from pysph.examples.sphysics.beach_geometry import get_beach_geometry_2d
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, LiuFluidForce,
                                  RigidBodyMotion, RK2StepRigidBody)


def get_wavespaddle_geometry(hdx=1.5, dx_f=0.1, dx_s=0.05, r_f=100., r_s=100.,
                             length=3.75, height=0.3, flat_l=1., angle=4.2364,
                             h_fluid=0.2, obstacle_side=0.06):
    x1, y1, x2, y2 = get_beach_geometry_2d(dx_s, length, height, flat_l,
                                           angle, 3)
    r1 = np.ones_like(x1) * r_s
    m1 = r1 * dx_s * dx_s
    h1 = np.ones_like(x1) * hdx * dx_s
    cs1 = np.zeros_like(x1)
    rad1 = np.ones_like(x1) * dx_s
    wall = get_particle_array(
        name='wall', x=x1, y=y1, rho=r1, m=m1, h=h1, cs=cs1, rad_s=rad1)
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
    cs3 = np.zeros_like(x3)
    rad3 = np.ones_like(x3) * dx_f
    fluid = get_particle_array(
        name='fluid', x=x3, y=y3, rho=r3, m=m3, h=h3)
    square_center = np.array([-0.38, 0.16])
    x4, y4 = get_2d_block(dx_s, obstacle_side, obstacle_side, square_center)
    b1 = np.zeros_like(x4, dtype=int)
    square_center = np.array([-0.7, 0.16])
    x5, y5 = get_2d_block(dx_s, obstacle_side, obstacle_side, square_center)
    b2 = np.ones_like(x5, dtype=int)
    square_center = np.array([-1.56, 0.22])
    x6, y6 = get_2d_block(dx_s, obstacle_side, obstacle_side, square_center)
    b3 = np.ones_like(x5, dtype=int) * 2
    b = np.concatenate([b1, b2, b3])
    x4 = np.concatenate([x4, x5, x6])
    y4 = np.concatenate([y4, y5, y6])
    r4 = np.ones_like(x4) * r_s * 0.5
    m4 = r4 * dx_s * dx_s
    h4 = np.ones_like(x4) * hdx * dx_s
    cs4 = np.zeros_like(x4)
    rad4 = np.ones_like(x4) * dx_s
    obstacle = get_particle_array_rigid_body(
        name='obstacle', x=x4, y=y4, h=h4, rho=r4, m=m4, cs=cs4,
        rad_s=rad4, body_id=b)
    remove_overlap_particles(fluid, wall, dx_s, 2)
    remove_overlap_particles(fluid, paddle, dx_s, 2)
    remove_overlap_particles(fluid, obstacle, dx_s, 2)
    return fluid, wall, paddle, obstacle


class WavesPaddle2D(Application):

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.3,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.01,
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
        paddle.u = amplitude * (paddle.y - self.dx) * np.cos(theta)
        paddle.v = amplitude * (flat_l - paddle.x) * np.cos(theta)

    def create_particles(self):
        f, w, pad, obst = get_wavespaddle_geometry(
            self.hdx, self.dx, 0.75 * self.dx, length=lx, height=ly,
            h_fluid=h_fluid, obstacle_side=side, flat_l=flat_l,
            r_f=ro, r_s=ro)
        self.scheme.setup_properties([f, w, pad, obst], clean=False)
        for p in ['u0', 'v0', 'w0', 'x0', 'y0', 'z0']:
            pad.add_property(p)
        particles = [f, w, pad, obst]
        return particles

    def create_scheme(self):
        wcsph = WCSPHScheme(['fluid'], ['wall', 'paddle', 'obstacle'], dim=2,
                            rho0=ro, c0=co, h0=0.01, hdx=1.3, gy=-9.81,
                            hg_correction=True, alpha=alp, gamma=gamma,
                            update_h=True)
        edac = EDACScheme(['fluid'], ['wall', 'paddle', 'obstacle'], dim=2,
                          rho0=ro, c0=co, gy=-9.81, alpha=alp, nu=0.0, h=0.01,
                          clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, edac=edac)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqn1 = Group(equations=[
            BodyForce(dest='obstacle', sources=None, gy=-9.81),
            RigidBodyCollision(dest='obstacle', sources=['wall'],
                               kn=1.0e4, en=0.8)], real=False)
        eqn2 = Group(equations=[
            LiuFluidForce(dest='fluid', sources=['obstacle'])])
        eqn3 = Group(equations=[
            RigidBodyMoments(dest='obstacle', sources=None)])
        eqn4 = Group(equations=[
            RigidBodyMotion(dest='obstacle', sources=None)])
        eqns.append(eqn1)
        eqns.append(eqn2)
        eqns.append(eqn3)
        eqns.append(eqn4)
        return eqns

    def configure_scheme(self):
        s = self.scheme
        scheme = self.options.scheme
        if scheme == 'wcsph':
            s.configure(h0=self.h0, hdx=self.hdx)
        elif scheme == 'edac':
            s.configure(h=self.h0)
        step = dict(paddle=TwoStageRigidBodyStep(),
                    obstacle=RK2StepRigidBody())
        s.configure_solver(
            kernel=CubicSpline(dim=2), tf=7.0, dt=self.dt,
            adaptive_timestep=False, extra_steppers=step)


if __name__ == '__main__':
    h_fluid = 0.18
    co = 10.0 * np.sqrt(2.0 * 9.81 * h_fluid)
    ro = 1000.0
    alp = 0.2
    gamma = 7.0
    flat_l = 2.0
    side = 0.06
    lx = 4.75
    ly = 0.3
    amplitude = 1.5
    period = 1.4
    app = WavesPaddle2D()
    app.run()
