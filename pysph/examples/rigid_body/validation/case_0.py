"""A cube translating and rotating freely without the influence of gravity.

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
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


class Case0(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 3

    def create_scheme(self):
        # rbss = RigidBodySimpleScheme
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=None, dim=3,
                                     kn=self.kn, mu=self.mu, en=self.en)
        rbrms = RigidBodyRotationMatricesScheme(bodies=['body'], solids=None,
                                                dim=3, kn=self.kn, mu=self.mu,
                                                en=self.en)
        rbqs = RigidBodyQuaternionScheme(bodies=['body'], solids=None, dim=3,
                                         kn=self.kn, mu=self.mu, en=self.en)
        rbrmos = RigidBodyRotationMatricesOptimizedScheme(
            bodies=['body'], solids=None, dim=3, kn=self.kn, mu=self.mu,
            en=self.en)
        rbqos = RigidBodyQuaternionsOptimizedScheme(
            bodies=['body'], solids=None, dim=3, kn=self.kn, mu=self.mu,
            en=self.en)
        rbrmcs = RigidBodyRotationMatricesCompyleScheme(
            bodies=['body'], solids=None, dim=3, kn=self.kn, mu=self.mu,
            en=self.en)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs,
                          rbrmos=rbrmos, rbqos=rbqos, rbrmcs=rbrmcs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 5.
        dt = 1e-3
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = self.dx
        x, y, z = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j, 0:1:nz * 1j]
        x = x.flat
        y = y.flat
        z = (z - 1).flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body(name='body', x=x, y=y, z=z, h=h,
                                             m=m, rad_s=rad_s)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=x, y=y, z=z, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=x, y=y, z=z, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbrmos':
            body = get_particle_array_rigid_body_rotation_matrix_optimized(
                name='body', x=x, y=y, z=z, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqos':
            body = get_particle_array_rigid_body_quaternion_optimized(
                name='body', x=x, y=y, z=z, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        if self.options.scheme == 'rbrmcs':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=x, y=y, z=z, h=h, m=m, rad_s=rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        body.vc[0] = 0.5
        body.vc[1] = 0.5
        body.omega[2] = 1.
        return [body]


if __name__ == '__main__':
    app = Case0()
    app.run()
