"""
Simulation of solid-fluid mixture flow using moving particle methods
Shuai Zhang

link: https://www.sciencedirect.com/science/article/pii/S0021999108006499

Time: 7 minutes
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce)
from pysph.sph.rb_rotation_matrices import (
    get_particle_array_rigid_body, SumUpExternalForces, RK2StepRigidBody,
    RigidBodyCollision)
from pysph.tools.geometry import (get_2d_circle, get_2d_tank)


class ZhangStackOfCylinders(Application):
    def initialize(self):
        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = 1e-3
        self.dam_layers = 2
        self.dam_rho = 2000.

        self.cylinder_radius = 1. / 2. * 1e-2
        self.cylinder_diameter = 1. * 1e-2
        self.cylinder_spacing = 1e-3
        self.cylinder_rho = 2000.

        self.wall_height = 20 * 1e-2
        self.wall_spacing = 1e-3
        self.wall_layers = 2
        self.wall_time = 0.3
        self.wall_rho = 2000.

        # simulation properties
        self.hdx = 1.2
        self.alpha = 0.1

        # solver data
        self.tf = 0.6 + self.wall_time
        self.dt = 5e-5

    def create_particles(self):
        # get bodyid for each cylinder
        xc, yc, body_id = self.create_cylinders_stack()
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_radius
        rad_s = self.cylinder_spacing / 2.
        V = self.cylinder_spacing**2
        cylinders = get_particle_array_rigid_body(
            x=xc, y=yc, h=h, m=m, rho=self.cylinder_rho, rad_s=rad_s, V=V,
            body_id=body_id, name="cylinders")

        xd, yd = self.create_dam()
        dam = get_particle_array_rigid_body(
            x=xd, y=yd, h=h, m=m, rho=self.dam_rho,
            rad_s=self.dam_spacing / 2., V=self.dam_spacing**2, name="dam")

        xw, yw = self.create_wall()
        wall = get_particle_array_rigid_body(
            x=xw, y=yw, h=h, m=m, rho=self.wall_rho,
            rad_s=self.wall_spacing / 2., V=self.wall_spacing**2, name="wall")
        return [cylinders, dam, wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(cylinders=RK2StepRigidBody())

        dt = self.dt
        print("DT: %s" % dt)
        tf = self.tf
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf)

        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    BodyForce(dest='cylinders', sources=None, gy=-9.81),
                ], real=False),
            Group(equations=[
                RigidBodyCollision(dest='cylinders', sources=[
                    'dam', 'wall', 'cylinders'
                ], kn=1e7, en=0.5),
            ]),
            Group(equations=[
                SumUpExternalForces(dest='cylinders', sources=None)
            ]),
        ]
        return equations

    def create_dam(self):
        xt, yt = get_2d_tank(self.dam_spacing,
                             np.array([self.dam_length / 2., 0.]),
                             length=self.dam_length, height=self.dam_height,
                             num_layers=self.dam_layers, outside=True)

        return xt, yt

    def create_wall(self):
        x = np.arange(0.067, 0.067 + 2 * self.wall_spacing, self.wall_spacing)
        y = np.arange(0., self.wall_height, self.wall_spacing)
        xw, yw = np.meshgrid(x, y)
        return xw.ravel(), yw.ravel()

    def create_cylinders_stack(self):
        # create a stack of cylinders

        # first crate all six cylinder rows
        # first six set of cylinders
        xc1, yc1 = get_2d_circle(self.cylinder_spacing, self.cylinder_radius, [
            self.cylinder_radius + self.cylinder_spacing,
            self.cylinder_radius + self.cylinder_spacing
        ])

        xc2, yc2 = (xc1 + self.cylinder_diameter + self.cylinder_spacing, yc1)
        xc3, yc3 = (xc2 + self.cylinder_diameter + self.cylinder_spacing, yc2)
        xc4, yc4 = (xc3 + self.cylinder_diameter + self.cylinder_spacing, yc3)
        xc5, yc5 = (xc4 + self.cylinder_diameter + self.cylinder_spacing, yc4)
        xc6, yc6 = (xc5 + self.cylinder_diameter + self.cylinder_spacing, yc5)

        x_six_1, y_six_1 = np.concatenate(
            (xc1, xc2, xc3, xc4, xc5, xc6)), np.concatenate((yc1, yc2, yc3,
                                                             yc4, yc5, yc6))
        x_six_2, y_six_2 = x_six_1, y_six_1 + 2. * self.cylinder_diameter
        x_six_3, y_six_3 = x_six_2, y_six_2 + 2. * self.cylinder_diameter

        x_six, y_six = np.concatenate(
            (x_six_1, x_six_2, x_six_3)), np.concatenate((y_six_1, y_six_2,
                                                          y_six_3))

        # now create five cylinder rows
        xc1, yc1 = get_2d_circle(self.cylinder_spacing, self.cylinder_radius, [
            self.cylinder_diameter + self.cylinder_spacing,
            3. * self.cylinder_radius + self.cylinder_spacing
        ])
        xc2, yc2 = (xc1 + self.cylinder_diameter + self.cylinder_spacing, yc1)
        xc3, yc3 = (xc2 + self.cylinder_diameter + self.cylinder_spacing, yc2)
        xc4, yc4 = (xc3 + self.cylinder_diameter + self.cylinder_spacing, yc3)
        xc5, yc5 = (xc4 + self.cylinder_diameter + self.cylinder_spacing, yc4)

        x_five_1, y_five_1 = np.concatenate(
            (xc1, xc2, xc3, xc4, xc5)), np.concatenate((yc1, yc2, yc3, yc4,
                                                        yc5))
        x_five_2, y_five_2 = x_five_1, y_five_1 + 2. * self.cylinder_diameter
        x_five_3, y_five_3 = x_five_2, y_five_2 + 2. * self.cylinder_diameter
        x_five, y_five = np.concatenate((x_five_1, x_five_2,
                                         x_five_3)), np.concatenate(
                                             (y_five_1, y_five_2, y_five_3))

        x, y = np.concatenate((x_six, x_five)), np.concatenate((y_six, y_five))

        # create body_id
        no_particles_one_cylinder = len(xc1)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def geometry(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        # please run this function to know how
        # geometry looks like
        xc, yc, body_id = self.create_cylinders_stack()
        xt, yt = self.create_dam()
        xw, yw = self.create_wall()

        plt.scatter(xc, yc)
        plt.scatter(xt, yt)
        plt.scatter(xw, yw)
        plt.axes().set_aspect('equal', 'datalim')
        print("done")
        plt.show()

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'wall':
                    pa.y += 14 * 1e-2

    def post_process(self):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.

        By running the example it becomes much clear.
        """
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        files = self.output_files
        print(len(files))
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files, 'cylinders'):
            _t = sd['t']
            if _t > self.wall_time:
                t.append(_t)
                # get the system center
                cm_x = 0
                cm_y = 0
                for i in range(array.n_body[0]):
                    cm_x += array.xcm[3 * i] * array.total_mass[i]
                    cm_y += array.xcm[3 * i + 1] * array.total_mass[i]
                cm_x = cm_x / np.sum(array.total_mass)
                cm_y = cm_y / np.sum(array.total_mass)

                system_x.append(cm_x / self.dam_length)
                system_y.append(cm_y / self.dam_length)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        t = t - np.min(t)

        plt.plot(t, system_x, label='system com x')
        plt.plot(t, system_y, label='system com y')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    app.run()
    app.post_process()
