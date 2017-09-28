"""
SPHysics case8 - dambreak with obstacles (30 minutes)
"""

from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator_step import TwoStageRigidBodyStep
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array, get_particle_array_rigid_body

import numpy as np
from pysph.sph.equation import Group, Equation
from pysph.sph.scheme import AdamiHuAdamsScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.geometry import get_2d_block, get_2d_wall
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody, LiuFluidForce)
from pysph.examples.sphysics.periodic_rigidbody import GroupParticles


def get_geometry(dx_s=0.03, dx_f=0.03, hdx=1.3, r_f=100.0, r_s=100.0,
                 wall_l=4.0, wall_h=2.0, fluid_l=1., fluid_h=2., cube_s=0.25):
    wall_y1 = np.arange(dx_s, wall_h, dx_s)
    wall_xlayer = np.ones_like(wall_y1) * 2.0
    wall_x1 = []
    wall_x2 = []
    num_layers = 3
    for i in range(num_layers):
        wall_x1.append(wall_xlayer + i * dx_s)
        wall_x2.append(wall_xlayer - i * dx_s + wall_l / 4.0)
    wall_x1, wall_x2 = np.ravel(wall_x1), np.ravel(wall_x2)
    wall_y1 = np.tile(wall_y1, num_layers)
    wall_y2 = wall_y1
    w_center = np.array([wall_l / 2.0, 0.0])
    wall_x3, wall_y3 = get_2d_wall(dx_s, w_center, wall_l, num_layers, False)
    w_center = np.array([2.5, wall_h + dx_s / 2.0])
    wall_x4, wall_y4 = get_2d_wall(dx_s, w_center, 1.0, num_layers)
    wall_x = np.concatenate([wall_x1, wall_x2, wall_x3, wall_x4])
    wall_y = np.concatenate([wall_y1, wall_y2, wall_y3, wall_y4])
    r1 = np.ones_like(wall_x) * r_s
    m1 = r1 * dx_s * dx_s
    h1 = np.ones_like(wall_x) * dx_s * hdx
    cs1 = np.zeros_like(wall_x)
    rad1 = np.ones_like(wall_x) * dx_s
    wall = get_particle_array(name='wall', x=wall_x, y=wall_y, h=h1, rho=r1,
                              m=m1, cs=cs1, rad_s=rad1)
    f_center = np.array([3.0 * wall_l / 8.0, wall_h / 2.0])
    x2, y2 = get_2d_block(dx_f, fluid_l, fluid_h, f_center)
    r2 = np.ones_like(x2) * r_f
    m2 = r2 * dx_f * dx_f
    h2 = np.ones_like(x2) * dx_f * hdx
    cs2 = np.zeros_like(x2)
    rad2 = np.ones_like(x2) * dx_f
    fluid = get_particle_array(name='fluid', x=x2, y=y2, h=h2, rho=r2, m=m2,
                               cs=cs2, rad_s=rad2)
    center1 = np.array([wall_l / 8.0 + cube_s / 2.0,
                        wall_h / 4.0 + cube_s / 2.0])
    cube1_x, cube1_y = get_2d_block(dx_s, cube_s, cube_s, center1)
    b1 = np.zeros_like(cube1_x, dtype=int)
    center2 = np.array([3.0 * wall_l / 4.0 + cube_s / 2.0 + 3.0 * dx_s,
                        wall_h + cube_s / 2.0 + (num_layers + 1) * dx_s])
    cube2_x, cube2_y = get_2d_block(dx_s, cube_s, cube_s, center2)
    b2 = np.ones_like(cube2_x, dtype=int)
    b = np.concatenate([b1, b2])
    x3 = np.concatenate([cube1_x, cube2_x])
    y3 = np.concatenate([cube1_y, cube2_y])
    r3 = np.ones_like(x3) * r_s * 0.5
    m3 = r3 * dx_s * dx_s
    h3 = np.ones_like(x3) * dx_s * hdx
    cs3 = np.zeros_like(x3)
    rad3 = np.ones_like(x3) * dx_s
    cube = get_particle_array_rigid_body(
        name='cube', x=x3, y=y3, h=h3, cs=cs3, rho=r3, m=m3, rad_s=rad3,
        body_id=b)
    remove_overlap_particles(fluid, wall, dx_s, 2)
    return fluid, wall, cube


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
        self.dt = 0.25 * self.h0 / co

    def create_domain(self):
        return DomainManager(xmin=0.0, xmax=4.0, periodic_in_x=True)

    def create_particles(self):
        fluid, wall, cube = get_geometry(0.5 * self.dx, self.dx, self.hdx)
        self.scheme.setup_properties([fluid, wall, cube], clean=False)
        for p in ['u0', 'v0', 'w0', 'x0', 'y0', 'z0']:
            wall.add_property(p)
        for p in ['fx', 'fy', 'fz', 'V', 'arho']:
            cube.add_property(p)
        particles = [fluid, wall, cube]
        return particles

    def create_scheme(self):
        wcsph = WCSPHScheme(['fluid'], ['wall', 'cube'], dim=2, rho0=ro,
                            h0=0.03, hdx=1.3, hg_correction=True, c0=co,
                            gy=-9.81, alpha=alp, gamma=gamma, update_h=True)
        edac = EDACScheme(['fluid'], ['wall', 'cube'], dim=2, rho0=ro, c0=co,
                          alpha=alp, nu=0.0, h=0.03, gy=-9.81, clamp_p=True)
        return SchemeChooser(default='wcsph', wcsph=wcsph, edac=edac)

    def configure_scheme(self):
        s = self.scheme
        scheme = self.options.scheme
        if scheme == 'wcsph':
            s.configure(h0=self.h0, hdx=self.hdx)
        elif scheme == 'edac':
            s.configure(h=self.h0)
        step = dict(cube=RK2StepRigidBody())
        s.configure_solver(kernel=CubicSpline(dim=2), dt=self.dt, tf=3.0,
                           adaptive_timestep=False, extra_steppers=step)

    def create_equations(self):
        eqns = self.scheme.get_equations()
        eqn1 = Group(equations=[
            BodyForce(dest='cube', sources=None, gy=-9.81),
            RigidBodyCollision(dest='cube', sources=['wall', 'cube'],
                               kn=1.0e5, en=0.8),
            LiuFluidForce(dest='fluid', sources=['cube'])], real=False)
        eqn2 = Group(equations=[
            GroupParticles('cube', xmin=0.0, xmax=4.0, periodic_in_x=True)],
            real=False)
        eqn3 = Group(equations=[
            RigidBodyMoments(dest='cube', sources=None)], real=False)
        eqn4 = Group(equations=[
            RigidBodyMotion(dest='cube', sources=None)], real=False)
        eqns.append(eqn1)
        eqns.append(eqn2)
        eqns.append(eqn3)
        eqns.append(eqn4)
        return eqns


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
