"""A 2d cube bouncing inside a box. (5 seconds)

This is used to test the rigid body equations.
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
    get_particle_array_rigid_body_rotation_matrix,
    get_particle_array_rigid_body_quaternion,
    get_particle_array_rigid_body_rotation_matrix_optimized)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_2d_tank


class Case3(Application):
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
        nx, ny = 10, 10
        dx = 1.0 / (nx - 1)
        x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
        x = x.flat
        y = y.flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body(name='body', x=x, y=y, h=h, m=m,
                                             rad_s=rad_s)

        body.vc[0] = -3.0
        body.vc[1] = -3.0
        body.omega[2] = 1.0

        # Create the tank.
        x, y = get_2d_tank(dx, base_center=[1., -2.], length=5.0, height=5.0,
                           num_layers=3)
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        tank = get_particle_array_rigid_body(name='tank', x=x, y=y, h=h, m=m,
                                             rad_s=rad_s)
        tank.total_mass[0] = np.sum(m)

        if self.options.scheme == 'rbrms':
            body = get_particle_array_rigid_body_rotation_matrix(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')

        elif self.options.scheme == 'rbqs':
            body = get_particle_array_rigid_body_quaternion(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')
        elif self.options.scheme == 'rbrmos':
            body = get_particle_array_rigid_body_rotation_matrix_optimized(
                name='body', x=body.x, y=body.y, h=body.h, m=body.m,
                rad_s=body.rad_s)
            add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                           'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                           'tang_disp_z')
        body.vc[0] = -3.0
        body.vc[1] = -3.0
        body.omega[2] = 1.0
        return [body, tank]

    def create_scheme(self):
        # rbss = RigidBodySimpleScheme
        rbss = RigidBodySimpleScheme(bodies=['body'], solids=['tank'],
                                     dim=self.dim, kn=self.kn, mu=self.mu,
                                     en=self.en, gy=-9.81)
        rbrms = RigidBodyRotationMatricesScheme(bodies=['body'], solids=[
            'tank'
        ], dim=self.dim, kn=self.kn, mu=self.mu, en=self.en, gy=-9.81)
        rbqs = RigidBodyQuaternionScheme(bodies=['body'], solids=['tank'],
                                         dim=self.dim, kn=self.kn, mu=self.mu,
                                         en=self.en, gy=-9.81)
        rbrmos = RigidBodyRotationMatricesOptimizedScheme(
            bodies=['body'], solids=['tank'], dim=self.dim, kn=self.kn,
            mu=self.mu, en=self.en, gy=-9.81)
        s = SchemeChooser(default='rbss', rbss=rbss, rbrms=rbrms, rbqs=rbqs,
                          rbrmos=rbrmos)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 3.
        dt = 5e-4
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['tank']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.1
        viewer.scalar = 'u'

        b = particle_arrays['body']
        viewer.scalar = 'u'
        b.show_legend = True
        ''')

    # def post_process(self):
    #     if len(self.output_files) == 0:
    #         return

    #     from pysph.solver.utils import iter_output
    #     t, cm = [], []
    #     files = self.output_files
    #     for sd, array in iter_output(files, 'body'):
    #         t.append(sd['t'])
    #         cm.append(array.cm[0])
    #     import os
    #     import matplotlib.pyplot as plt

    #     plt.plot(t, cm)
    #     fig = os.path.join(self.output_dir, 't_vs_cm.png')
    #     plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Case3()
    app.run()
    # app.post_process()
