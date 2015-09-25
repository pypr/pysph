"""Very simple rigid body motion. (5 seconds)

This is used to test the rigid body equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import RigidBodyMoments, RigidBodyMotion, RK2StepRigidBody

dim = 3

dt = 1e-3
tf = 2.5

hdx = 1.0
rho0 = 10.0


class SimpleRigidMotion(Application):
    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = 1.0/(nx-1)
        x, y, z = np.mgrid[0:1:nx*1j, 0:1:ny*1j, 0:1:nz*1j]
        x = x.flat
        y = y.flat
        z = z.flat
        m = np.ones_like(x)*dx*dx*rho0
        h = np.ones_like(x)*hdx*dx
        body = get_particle_array_rigid_body(
            name='body', x=x, y=y, z=z, h=h, m=m,
        )

        body.omega[0] = 5.0
        body.omega[1] = 5.0
        body.vc[0] = 1.0
        body.vc[1] = 1.0

        return [body]

    def create_solver(self):
        kernel = CubicSpline(dim=dim)
        integrator = EPECIntegrator(body=RK2StepRigidBody())
        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=False)
        solver.set_print_freq(10)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[RigidBodyMoments(dest='body', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='body', sources=None)]),
        ]
        return equations

if __name__ == '__main__':
    app = SimpleRigidMotion()
    app.run()
