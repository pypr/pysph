"""A sphere of density 500 falling into a hydrostatic tank (15 minutes)

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, SummationDensity)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    BodyForce, SummationDensityBoundary, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


def create_boundary():
    dx = 2

    # bottom particles in tank
    xb = np.arange(-2 * dx, 100 + 2 * dx, dx)
    yb = np.arange(-2 * dx, 0, dx)
    xb, yb = np.meshgrid(xb, yb)
    xb = xb.ravel()
    yb = yb.ravel()

    xl = np.arange(-2 * dx, 0, dx)
    yl = np.arange(0, 250, dx)
    xl, yl = np.meshgrid(xl, yl)
    xl = xl.ravel()
    yl = yl.ravel()

    xr = np.arange(100, 100 + 2 * dx, dx)
    yr = np.arange(0, 250, dx)
    xr, yr = np.meshgrid(xr, yr)
    xr = xr.ravel()
    yr = yr.ravel()

    x = np.concatenate([xl, xb, xr])
    y = np.concatenate([yl, yb, yr])

    return x * 1e-3, y * 1e-3


def create_fluid():
    dx = 2
    xf = np.arange(0, 100, dx)
    yf = np.arange(0, 150, dx)
    xf, yf = np.meshgrid(xf, yf)
    xf = xf.ravel()
    yf = yf.ravel()

    return xf * 1e-3, yf * 1e-3


def create_sphere(dx=1):
    x = np.arange(0, 100, dx)
    y = np.arange(151, 251, dx)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()

    p = ((x - 50)**2 + (y - 200)**2) < 20**2
    x = x[p]
    y = y[p]

    # lower sphere a little
    y = y - 20
    return x * 1e-3, y * 1e-3


def get_density(y):
    height = 150
    c_0 = 2 * np.sqrt(2 * 9.81 * height * 1e-3)
    rho_0 = 1000
    height_water_clmn = height * 1e-3
    gamma = 7.
    _tmp = gamma / (rho_0 * c_0**2)

    rho = np.zeros_like(y)
    for i in range(len(rho)):
        p_i = rho_0 * 9.81 * (height_water_clmn - y[i])
        rho[i] = rho_0 * (1 + p_i * _tmp)**(1. / gamma)
    return rho


def geometry():
    import matplotlib.pyplot as plt
    # please run this function to know how
    # geometry looks like
    x_tank, y_tank = create_boundary()
    x_fluid, y_fluid = create_fluid()
    x_cube, y_cube = create_sphere()
    plt.scatter(x_fluid, y_fluid)
    plt.scatter(x_tank, y_tank)
    plt.scatter(x_cube, y_cube)
    plt.axes().set_aspect('equal', 'datalim')
    print("done")
    plt.show()


class RigidFluidCoupling(Application):
    def initialize(self):
        self.dx = 2 * 1e-3
        self.hdx = 1.2
        self.ro = 1000
        self.solid_rho = 500
        self.m = 1000 * self.dx * self.dx
        self.co = 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
        self.alpha = 0.1

    def create_particles(self):
        """Create the circular patch of fluid."""
        xf, yf = create_fluid()
        m = self.ro * self.dx * self.dx
        rho = self.ro
        h = self.hdx * self.dx
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        dx = 2
        xt, yt = create_boundary()
        m = 1000 * self.dx * self.dx
        rho = 1000
        rad_s = 2 / 2. * 1e-3
        h = self.hdx * self.dx
        V = dx * dx * 1e-6
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        rad_s=rad_s, V=V, name="tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        dx = 1
        xc, yc = create_sphere(1)
        m = self.solid_rho * dx * 1e-3 * dx * 1e-3
        rho = self.solid_rho
        h = self.hdx * self.dx
        rad_s = dx / 2. * 1e-3
        V = dx * dx * 1e-6
        cs = 0.0
        cube = get_particle_array_rigid_body(x=xc, y=yc, h=h, m=m, rho=rho,
                                             rad_s=rad_s, V=V, cs=cs,
                                             name="cube")

        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), tank=WCSPHStep(),
                                    cube=RK2StepRigidBody())

        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        # dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(
            kernel=kernel,
            dim=2,
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
                SummationDensity(
                    dest='fluid',
                    sources=['fluid'], ),
                SummationDensityBoundary(
                    dest='fluid', sources=['tank', 'cube'], fluid_rho=1000.0)
            ]),

            # Tait equation of state
            Group(equations=[
                TaitEOSHGCorrection(dest='fluid', sources=None, rho0=self.ro,
                                    c0=self.co, gamma=7.0),
            ], real=False),
            Group(equations=[
                MomentumEquation(dest='fluid', sources=['fluid'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                AkinciRigidFluidCoupling(dest='fluid',
                                         sources=['cube', 'tank']),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
            Group(equations=[
                RigidBodyCollision(dest='cube', sources=['tank'], kn=1e5)
            ]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
