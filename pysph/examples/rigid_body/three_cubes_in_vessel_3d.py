"""Three cubes of different density falling into a static tank of
water in 3d.(20 minutes)

An example to illustrate 3d pysph framework
"""
from __future__ import print_function
import numpy as np

from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    BodyForce, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


def get_3d_dam(length=10, height=15, depth=10, dx=0.1, layers=2):
    _x = np.arange(0, length, dx)
    _y = np.arange(0, height, dx)
    _z = np.arange(0, depth, dx)

    x, y, z = np.meshgrid(_x, _y, _z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()

    # get particles inside the tank
    tmp = layers - 1
    cond_1 = (x > tmp * dx) & (x < _x[-1] - tmp * dx) & (y > tmp * dx)

    cond_2 = (z > tmp * dx) & (z < z[-1] - tmp * dx)

    cond = cond_1 & cond_2
    # exclude inside particles
    x, y, z = x[~cond], y[~cond], z[~cond]

    return x, y, z


def get_3d_block(length=10, height=15, depth=10, dx=0.1):
    x = np.arange(0, length, dx)
    y = np.arange(0, height, dx)
    z = np.arange(0, depth, dx)

    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    return x, y, z


def get_fluid_and_dam_geometry_3d(d_l, d_h, d_d, f_l, f_h, f_d, d_layers, d_dx,
                                  f_dx, fluid_left_extreme=None,
                                  tank_outside=False):
    xd, yd, zd = get_3d_dam(d_l, d_h, d_d, d_dx, d_layers)
    xf, yf, zf = get_3d_block(f_l, f_h, f_d, f_dx)

    if fluid_left_extreme:
        x_trans, y_trans, z_trans = fluid_left_extreme
        xf += x_trans
        yf += y_trans
        zf += z_trans

    else:
        xf += 2 * d_dx
        yf += 2 * d_dx
        zf += 2 * d_dx

    return xd, yd, zd, xf, yf, zf


def get_sphere(centre=[0, 0, 0], radius=1, dx=0.1):
    x = np.arange(0, radius * 2, dx)
    y = np.arange(0, radius * 2, dx)
    z = np.arange(0, radius * 2, dx)

    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()

    cond = ((x - radius)**2 + (y - radius)**2) + (z - radius)**2 <= radius**2

    x, y, z = x[cond], y[cond], z[cond]

    x_trans = centre[0] - radius
    y_trans = centre[1] - radius
    z_trans = centre[2] - radius

    x = x + x_trans
    y = y + y_trans
    z = z + z_trans

    return x, y, z


class RigidFluidCoupling(Application):
    def initialize(self):
        self._spacing = 4
        self.spacing = self._spacing * 1e-3
        self.dx = self.spacing
        self.hdx = 1.2
        self.ro = 1000
        self.solid_rho = 800
        self.m = 1000 * self.dx * self.dx * self.dx
        self.co = 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
        self.alpha = 0.1

    def create_particles(self):
        # get coordinates of tank and fluid
        tank_len = 150
        tank_hei = 150
        tank_dep = 150
        layers = 2

        flu_len = 150 - 2 * layers * self._spacing
        flu_hei = 52
        flu_dep = 150 - 2 * layers * self._spacing

        xt, yt, zt, xf, yf, zf = get_fluid_and_dam_geometry_3d(
            d_l=tank_len, d_h=tank_hei, d_d=tank_dep, f_l=flu_len, f_h=flu_hei,
            f_d=flu_dep, d_layers=2, d_dx=self._spacing, f_dx=self._spacing)
        # scale it to mm
        xt, yt, zt, xf, yf, zf = (xt * 1e-3, yt * 1e-3, zt * 1e-3, xf * 1e-3,
                                  yf * 1e-3, zf * 1e-3)

        # get coordinates of cube
        xc, yc, zc = get_3d_block(20, 20, 20, self._spacing/2.)
        xc1, yc1, zc1 = ((xc + 60) * 1e-3, (yc + 120) * 1e-3, (zc + 70) * 1e-3)
        xc2, yc2, zc2 = ((xc + 4 * self._spacing) * 1e-3, (yc + 120) * 1e-3,
                         (zc + 70) * 1e-3)
        xc3, yc3, zc3 = ((xc + 100) * 1e-3, (yc + 120) * 1e-3,
                         (zc + 70) * 1e-3)

        xc, yc, zc = (np.concatenate((xc1, xc2, xc3)), np.concatenate(
            (yc1, yc2, yc3)), np.concatenate((zc1, zc2, zc3)))

        # Create particle array for fluid
        m = self.ro * self.spacing * self.spacing * self.spacing
        rho = self.ro
        h = self.hdx * self.spacing
        fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                         name="fluid")

        # Create particle array for tank
        m = 1000 * self.spacing**3
        rho = 1000
        rad_s = self.spacing / 2.
        h = self.hdx * self.spacing
        V = self.spacing**3
        tank = get_particle_array_wcsph(x=xt, y=yt, z=zt, h=h, m=m, rho=rho,
                                        rad_s=rad_s, V=V, name="tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        # Create particle array for cube
        h = self.hdx * self.spacing/2.

        # assign  density of three spheres
        rho1 = np.ones_like(xc1) * 2000
        rho2 = np.ones_like(xc1) * 800
        rho3 = np.ones_like(xc1) * 500
        rho = np.concatenate((rho1, rho2, rho3))

        # assign body id's
        body1 = np.ones_like(xc1, dtype=int) * 0
        body2 = np.ones_like(xc1, dtype=int) * 1
        body3 = np.ones_like(xc1, dtype=int) * 2
        body = np.concatenate((body1, body2, body3))

        m = rho * (self.spacing/2.)**3
        rad_s = self.spacing / 4.
        V = (self.spacing/2.)**3
        cs = 0.0
        cube = get_particle_array_rigid_body(x=xc, y=yc, z=zc, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs,
                                             body_id=body,
                                             name="cube")

        print(
            fluid.get_number_of_particles(),
            tank.get_number_of_particles(),
            cube.get_number_of_particles(), )
        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=3)

        integrator = EPECIntegrator(fluid=WCSPHStep(), tank=WCSPHStep(),
                                    cube=RK2StepRigidBody())

        # dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.6
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=dt,
            tf=tf,
            adaptive_timestep=False, )

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='cube', sources=None, gy=-9.81),
            ], real=False),
            Group(equations=[
                ContinuityEquation(dest='fluid',
                                   sources=['fluid', 'tank', 'cube']),
                ContinuityEquation(dest='tank',
                                   sources=['tank', 'fluid', 'cube'])
            ]),

            # Tait equation of state
            Group(equations=[
                TaitEOSHGCorrection(dest='fluid', sources=None, rho0=self.ro,
                                    c0=self.co, gamma=7.0),
                TaitEOSHGCorrection(dest='tank', sources=None, rho0=self.ro,
                                    c0=self.co, gamma=7.0),
            ], real=False),
            Group(equations=[
                MomentumEquation(dest='fluid', sources=['fluid', 'tank'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                AkinciRigidFluidCoupling(dest='fluid', sources=['cube']),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
            Group(equations=[
                RigidBodyCollision(dest='cube', sources=['tank', 'cube'],
                                   kn=1e5)
            ]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
