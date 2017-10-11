"""Ten sphere of different density falling into a hydrostatic tank
implemented using Akinci in 2d. (15 minutes)

Check basic equations of SPH to throw a ball inside the vessel

Geometry will be like




    ^  ||        ___      ___                            ___      ___       ||
    |  ||       /   \    /   \                          /   \    /   \      ||
    |  ||       |   |    |   |                          |   |    |   |      ||
    |  ||    ___\___/ ___\___/  ___                 ___ \___/ ___\___/ ___  ||
    |  ||   /   \    /   \     /   \               /   \     /   \    /   \ ||
    |  ||   |   |    |   |     |   |               |   |     |   |    |   | ||
    |  ||   \___/    \___/     \___/               \___/     \___/    \___/ ||
    |  ||___________________________________________________________________||
500mm  ||                          ^                                        ||
    |  ||                          |                                        ||
    |  ||                          |                                        ||
    |  ||                          |300 mm                                  ||
    |  ||                          |                                        ||
    |  ||                          |                                        ||
    |  ||                          |                                        ||
    v  ||__________________________v________________________________________||
       ||___________________________________________________________________||

       <--------------------------1000mm------------------------------------->






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
    BodyForce, RigidBodyCollision, SummationDensityBoundary, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


def get_2d_dam(length=10, height=15, dx=0.1, layers=2):
    _x = np.arange(0, length, dx)
    _y = np.arange(0, height, dx)

    x, y = np.meshgrid(_x, _y)
    x, y = x.ravel(), y.ravel()

    # get particles inside the tank
    cond = ((x > (layers - 1) * dx)) & ((x < (x[-1] - (layers - 1) * dx)) &
                                        (y > (layers - 1) * dx))

    # exclude inside particles
    x, y = x[~cond], y[~cond]

    return x, y


def get_2d_block(length=10, height=15, dx=0.1):
    x = np.arange(0, length, dx)
    y = np.arange(0, height, dx)

    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    return x, y


def get_fluid_and_dam_geometry(d_l, d_h, f_l, f_h, d_layers, d_dx, f_dx,
                               fluid_left_extreme=None):
    xd, yd = get_2d_dam(d_l, d_h, d_dx, d_layers)
    xf, yf = get_2d_block(f_l, f_h, f_dx)

    if fluid_left_extreme:
        x_trans, y_trans = fluid_left_extreme
        xf += x_trans
        yf += y_trans

    else:
        xf += 2 * d_dx
        yf += 2 * d_dx

    return xd, yd, xf, yf


def get_circle(centre=[0, 0], radius=1, dx=0.1):
    x = np.arange(0, radius * 2, dx)
    y = np.arange(0, radius * 2, dx)

    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    cond = ((x - radius)**2 + (y - radius)**2) <= radius**2

    x, y = x[cond], y[cond]

    x_trans = centre[0] - radius
    y_trans = centre[1] - radius

    x = x + x_trans
    y = y + y_trans

    return x, y


def create_ten_circles(radius=20 * 1e-3, spacing=1 * 1e-3,
                       fluid_height=300 * 1e-3):
    x1, y1 = get_circle(centre=[100 * 1e-3, fluid_height + radius + 30 * 1e-3],
                        radius=radius, dx=spacing)

    x2, y2 = x1 + 2 * radius, y1 + 3 * radius

    x3, y3 = x2 + 2 * radius, y1

    x4, y4 = x3 + 2 * radius, y2

    x5, y5 = x4 + 2 * radius, y3

    x_left = np.concatenate([x1, x2, x3, x4, x5])
    y_left = np.concatenate([y1, y2, y3, y4, y5])

    # x_middle, y_middle = x1 + 400 * 1e-3, y1 + 300 * 1e-3

    x_right = x_left + 500 * 1e-3
    y_right = y_left

    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    return x, y


def get_rho_of_each_sphere(xc, yc, radius=20 * 1e-3, spacing=1 * 1e-3):
    x1, y1 = get_circle(radius=radius, dx=spacing)
    pars = len(x1)

    rho = np.ones_like(xc)
    no_of_spheres = int(len(rho) / len(x1))

    for i in range(no_of_spheres):
        if i < 5:
            rho[i * pars:(i + 1) * pars] = 500
        if i >= 5:
            rho[i * pars:(i + 1) * pars] = 1500

    return rho


def get_body_id_of_each_sphere(xc, yc, radius=20 * 1e-3, spacing=1 * 1e-3):
    x1, y1 = get_circle(radius=radius, dx=spacing)
    pars = len(x1)

    body_id = np.ones_like(xc, dtype=int)
    no_of_spheres = int(len(body_id) / len(x1))

    for i in range(no_of_spheres):
        body_id[i * pars:(i + 1) * pars] = i

    return body_id


class RigidFluidCoupling(Application):
    def initialize(self):
        self.dam_length = 1000 * 1e-3
        self.dam_height = 500 * 1e-3
        self.dam_spacing = 2 * 1e-3
        self.dam_layers = 3

        self.fluid_length = (
            1000 * 1e-3 - 3 * self.dam_layers * self.dam_spacing)
        self.fluid_height = 300 * 1e-3
        self.fluid_spacing = 5 * 1e-3
        self.fluid_rho = 1000.

        self.sphere_radius = 30 * 1e-3
        self.sphere_spacing = 4 * 1e-3

        # simulation properties
        self.hdx = 1.2
        self.co = 2 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.alpha = 0.1

    def create_particles(self):
        # get the geometry
        xt, yt, xf, yf = get_fluid_and_dam_geometry(
            self.dam_length, self.dam_height, self.fluid_length,
            self.fluid_height, self.dam_layers, self.dam_spacing,
            self.fluid_spacing, [3 * self.dam_spacing, 3 * self.dam_spacing])

        # create fluid particle array
        m = self.fluid_rho * self.fluid_spacing * self.fluid_spacing
        rho = self.fluid_rho
        h = self.hdx * self.fluid_spacing
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        # create tank particle array
        m = self.fluid_rho * self.dam_spacing * self.dam_spacing
        rho = 1000
        rad_s = self.dam_spacing / 2.
        h = self.hdx * self.dam_spacing
        V = self.dam_spacing**2
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        rad_s=rad_s, V=V, name="tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        xc, yc = create_ten_circles(radius=self.sphere_radius,
                                    spacing=self.sphere_spacing,
                                    fluid_height=self.fluid_height)

        # get density of each sphere
        rho = get_rho_of_each_sphere(xc, yc, radius=self.sphere_radius,
                                     spacing=self.sphere_spacing)
        # get bodyid for each sphere
        body_id = get_body_id_of_each_sphere(xc, yc, radius=self.sphere_radius,
                                             spacing=self.sphere_spacing)
        m = rho * self.sphere_spacing**2
        h = self.hdx * self.sphere_spacing
        rad_s = self.sphere_spacing / 2.
        V = self.sphere_spacing**2
        cs = 0.0
        cube = get_particle_array_rigid_body(x=xc, y=yc, h=h, m=m, rho=rho,
                                             rad_s=rad_s, V=V, cs=cs,
                                             body_id=body_id, name="cube")
        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), cube=RK2StepRigidBody(),
                                    tank=WCSPHStep())

        dt = 1 * 1e-4
        print("DT: %s" % dt)
        tf = 1
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
                TaitEOSHGCorrection(dest='fluid', sources=None,
                                    rho0=self.fluid_rho, c0=self.co,
                                    gamma=7.0),
            ], real=False),
            Group(equations=[
                MomentumEquation(dest='fluid', sources=['fluid'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                AkinciRigidFluidCoupling(dest='fluid',
                                         sources=['cube', 'tank']),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank',
                                                      'cube']),
            ]),
            Group(equations=[
                RigidBodyCollision(dest='cube', sources=['tank', 'cube'],
                                   kn=1e5)
            ]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
        ]
        return equations

    def geometry(self):
        import matplotlib.pyplot as plt
        # please run this function to know how
        # geometry looks like
        x_tank, y_tank, x_fluid, y_fluid = get_fluid_and_dam_geometry(
            self.dam_length, self.dam_height, self.fluid_length,
            self.fluid_height, self.dam_layers, self.dam_spacing,
            self.fluid_spacing, [3 * self.dam_spacing, 3 * self.dam_spacing])

        x_cube, y_cube = create_ten_circles(radius=self.sphere_radius,
                                            spacing=self.sphere_spacing,
                                            fluid_height=self.fluid_height)
        plt.scatter(x_fluid, y_fluid)
        plt.scatter(x_tank, y_tank)
        plt.scatter(x_cube, y_cube)
        plt.axes().set_aspect('equal', 'datalim')
        print("done")
        plt.show()


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.geometry()
