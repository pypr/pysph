"""Many two dimensional cubes falling under gravity into a tank,
with hopper as an obstacle.

This is used to test the performance of rigid body dynamics schemes.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.scheme import SchemeChooser
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    RigidBodySimpleScheme, RigidBodyRotationMatricesScheme,
    RigidBodyQuaternionScheme, RigidBodyRotationMatricesOptimizedScheme,
    RigidBodyQuaternionsOptimizedScheme,
    RigidBodyRotationMatricesCompyleScheme,
    get_particle_array_rigid_body_rotation_matrix,
    get_particle_array_rigid_body_quaternion,
    get_particle_array_rigid_body_rotation_matrix_optimized,
    get_particle_array_rigid_body_quaternion_optimized)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_2d_tank
from pysph.examples.rigid_body.validation.case_2 import rotate_body


def create_one_body(dx):
    x, y = np.mgrid[0.:1:dx, 0.:1.:dx]
    return x.ravel(), y.ravel()


def create_three_bodies(dx):
    x1, y1 = create_one_body(dx)
    x2 = x1 + 1.1
    y2 = y1
    x3 = x2 + 1.1
    y3 = y2
    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    x = x + 2.
    return x, y


def create_many_bodies(dx, layers=10):
    x_tmp, y_tmp = create_three_bodies(dx)
    x = np.array([])
    y = np.array([])
    for i in range(layers):
        y_tmp = y_tmp + 1.1
        x = np.concatenate((x, x_tmp))
        y = np.concatenate((y, y_tmp))

    y = y - 3.9

    no_bodies = layers * 3
    x1, y1 = create_one_body(dx)
    bid_1 = np.ones_like(x1, dtype=int)
    bid = np.array([], dtype=int)
    for i in range(no_bodies):
        bid_tmp = bid_1 * i
        bid = np.concatenate((bid, bid_tmp))
    return x, y, bid


def create_2d_hopper(dx):
    x, y = np.mgrid[0:5:dx, 0:0.2:dx]
    x = x.ravel()
    y = y.ravel()
    x1, y1 = rotate_body(x, y, -60)
    x, y = np.mgrid[5:10:dx, 0:0.2:dx]
    x = x.ravel()
    y = y.ravel()
    x2, y2 = rotate_body(x, y, 60)
    x2 = x2 + 3.
    y2 = y2 - 8.6
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y


class Case4(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

    def create_particles(self):
        nx = 10
        dx = 1.0 / (nx - 1)
        xb, yb, bid = create_many_bodies(dx)
        m = np.ones_like(xb) * dx * dx * self.rho0
        h = np.ones_like(xb) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(xb) * dx / 2.
        body = get_particle_array_rigid_body(name='body', x=xb, y=yb, h=h, m=m,
                                             rad_s=rad_s, body_id=bid)

        # Create the tank.
        x, y = get_2d_tank(dx, base_center=[4, -8.], length=10.0, height=10.0,
                           num_layers=3)
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx / 2.
        tank = get_particle_array_rigid_body(name='tank', x=x, y=y, h=h, m=m,
                                             rad_s=rad_s)
        tank.total_mass[0] = np.sum(m)

        # Create hopper
        x, y = create_2d_hopper(dx)
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx / 2.
        hopper = get_particle_array_rigid_body(name='hopper', x=x, y=y, h=h,
                                               m=m, rad_s=rad_s)
        tank.total_mass[0] = np.sum(m)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbrmos':
            body = get_particle_array_rigid_body_rotation_matrix_optimized(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqos':
            body = get_particle_array_rigid_body_quaternion_optimized(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')
        elif self.options.scheme == 'rbrmcs':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s, body_id=body.body_id)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

            if body.backend == 'cython':
                from pysph.base.device_helper import DeviceHelper
                from compyle.api import get_config
                get_config().use_double = True
                body.set_device_helper(DeviceHelper(body))
                tank.set_device_helper(DeviceHelper(tank))
                hopper.set_device_helper(DeviceHelper(hopper))

        return [body, tank, hopper]

    def create_scheme(self):
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=[
            'tank', 'hopper'
        ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbrms = RigidBodyRotationMatricesScheme(bodies=['body'], solids=[
            'tank', 'hopper'
        ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbqs = RigidBodyQuaternionScheme(bodies=['body'], solids=[
            'tank', 'hopper'
        ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbrmos = RigidBodyRotationMatricesOptimizedScheme(
            bodies=['body'], solids=['tank', 'hopper'], dim=self.dim,
            kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbqos = RigidBodyQuaternionsOptimizedScheme(bodies=['body'], solids=[
            'tank', 'hopper'
        ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbrmcs = RigidBodyRotationMatricesCompyleScheme(
            bodies=['body'], solids=['tank', 'hopper'], dim=self.dim,
            kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs,
                          rbrmos=rbrmos, rbqos=rbqos, rbrmcs=rbrmcs)

        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        dt = 5e-4
        tf = 3.
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)


if __name__ == '__main__':
    app = Case4()
    app.run()
