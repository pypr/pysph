"""Four cubes bouncing inside a box. (10 seconds)

This is used to test the rigid body equations and the support for multiple
bodies.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
    RigidBodyMoments, RigidBodyMotion, RK2StepRigidBody)

dim = 3

dt = 5e-3
tf = 5.0
gz = -9.81

hdx = 1.0
rho0 = 10.0

def make_cube(lx, ly, lz, dx):
    """Return points x, y, z for a cube centered at origin with given lengths.
    """
    # Convert to floats to be safe with integer division.
    lx, ly, lz = list(map(float, (lx, ly, lz)))
    x, y, z = np.mgrid[-lx/2:lx/2+dx:dx,-ly/2:ly/2+dx:dx,-lz/2:lz/2+dx:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    return x, y, z


class BouncingCubes(Application):
    def create_particles(self):
        dx = 1.0/9.0
        _x, _y, _z = make_cube(0.5, 0.5, 0.5, dx)
        _z += 1.0
        _id = np.ones(_x.shape, dtype=int)
        x, y, z, body_id = [], [], [], []
        disp = [(0.4, 0, 0), (-0.4, 0, 0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
        for i, d in enumerate(disp):
            x.append(_x + d[0])
            y.append(_y + d[1])
            z.append(_z + d[2])
            body_id.append(_id*i)
        x = np.concatenate(x)
        y = np.concatenate(y)
        z = np.concatenate(z)
        body_id = np.concatenate(body_id)
        m = np.ones_like(x)*dx*dx*rho0
        h = np.ones_like(x)*hdx*dx

        # Split this one cube

        body = get_particle_array_rigid_body(
            name='body', x=x, y=y, z=z, h=h, m=m, body_id=body_id
        )

        body.vc[0] = 5.0
        body.vc[2] = -5.0
        body.vc[6] = -5.0
        body.vc[7] = -5.0
        body.vc[10] = 5.0

        # Create the tank.
        nx, ny, nz = 40, 40, 40
        dx = 1.0/(nx-1)
        xmin, xmax, ymin, ymax, zmin, zmax = -2, 2, -2, 2, -2, 2
        x, y, z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]
        interior = ((x < 1.8) & (x > -1.8)) & ((y < 1.8) & (y> -1.8)) & ((z > -1.8) & (z <= 2))
        tank = np.logical_not(interior)
        x = x[tank].flat
        y = y[tank].flat
        z = z[tank].flat
        m = np.ones_like(x)*dx*dx*rho0
        h = np.ones_like(x)*hdx*dx
        tank = get_particle_array_rigid_body(
            name='tank', x=x, y=y, z=z, h=h, m=m,
        )
        tank.total_mass[0] = np.sum(m)

        return [body, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=dim)

        integrator = EPECIntegrator(body=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                    dt=dt, tf=tf, adaptive_timestep=False)
        solver.set_print_freq(10)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='body', sources=None, gz=gz),
                RigidBodyCollision(
                    dest='body', sources=['tank', 'body'], k=1.0, d=2.0, eta=0.1, kt=0.1
                )]
            ),
            Group(equations=[RigidBodyMoments(dest='body', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='body', sources=None)]),
        ]
        return equations

if __name__ == '__main__':
    app = BouncingCubes()
    app.run()
