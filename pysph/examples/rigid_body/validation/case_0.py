"""A cube translating and rotating freely without the influence of gravity.

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.sph.rigid_body import (RigidBodySimpleScheme)
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
        s = SchemeChooser(default='rbss', rbss=rbss)
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

        body.vc[0] = 0.5
        body.vc[1] = 0.5
        body.omega[2] = 1.
        return [body]


if __name__ == '__main__':
    app = Case0()
    app.run()
