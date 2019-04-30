"""Simulating a cube bouncing inside a box with quaternion
approach . (23 seconds)

This is used to test the rigid body equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import BodyForce
from pysph.sph.rigid_body import (
    get_particle_array_rigid_body_quaternion, RigidBodyCollisionSimple,
    SumUpExternalForces, RK2StepRigidBodyQuaternions)
from pysph.tools.geometry import (get_2d_tank)

dim = 2

dt = 5e-4
tf = 10
gy = -9.81

hdx = 1.0
dx = dy = 0.02
rho0 = 10.0


class BouncingCube(Application):
    def create_particles(self):
        nx, ny = 10, 10
        dx = 1.0 / (nx - 1)
        x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
        x = x.ravel()
        y = y.ravel() + 2.
        m = np.ones_like(x) * dx * dx * rho0
        h = np.ones_like(x) * hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array_rigid_body_quaternion(
            name='body', x=x, y=y, h=h, m=m, rad_s=rad_s)
        # set the normals to the body
        maxx = max(x)
        fltr = (x == maxx)
        body.nx[fltr] = 1.
        body.nx0[fltr] = 1.

        body.vc[0] = -3.0
        body.vc[1] = -3.0

        # Create the tank.
        nx, ny = 10, 10
        dx = 1.0 / (nx - 1)
        xmin, xmax, ymin, ymax = -5, 5, -5, 5
        x, y = get_2d_tank(dx, np.array([0., 0.]), xmax - xmin, ymax - ymin, 3)

        m = np.ones_like(x) * dx * dx * rho0
        h = np.ones_like(x) * hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        tank = get_particle_array_rigid_body_quaternion(
            name='tank', x=x, y=y, h=h, m=m, rad_s=rad_s)
        tank.total_mass[0] = np.sum(m)

        return [body, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=dim)

        integrator = EPECIntegrator(body=RK2StepRigidBodyQuaternions())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='body', sources=None, gy=-9.81),
            ], real=False),
            Group(equations=[
                RigidBodyCollisionSimple(dest='body', sources=['tank'], kn=1e5,
                                         gamma_n=100.)
            ]),
            Group(equations=[SumUpExternalForces(dest='body', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = BouncingCube()
    app.run()
