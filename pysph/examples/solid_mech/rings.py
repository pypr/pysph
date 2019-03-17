"""Colliding Elastic Balls. (10 minutes)
"""

import numpy

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.solid_mech.basic import (get_particle_array_elastic_dynamics,
                                        ElasticSolidsScheme)

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import SolidMechStep


class Rings(Application):
    def initialize(self):
        # constants
        self.E = 1e7
        self.nu = 0.3975
        self.rho0 = 1.0

        self.dx = 0.0005
        self.hdx = 1.5
        self.h = self.hdx * self.dx

        # geometry
        self.ri = 0.03
        self.ro = 0.04

        self.spacing = 0.041

        self.dt = 1e-8
        self.tf = 5e-5

    def create_particles(self):
        spacing = self.spacing  # spacing = 2*5cm

        x, y = numpy.mgrid[-self.ro:self.ro:self.dx, -self.ro:self.ro:self.dx]
        x = x.ravel()
        y = y.ravel()

        d = (x * x + y * y)
        ro = self.ro
        ri = self.ri
        keep = numpy.flatnonzero((ri * ri <= d) * (d < ro * ro))
        x = x[keep]
        y = y[keep]

        x = numpy.concatenate([x - spacing, x + spacing])
        y = numpy.concatenate([y, y])

        dx = self.dx
        hdx = self.hdx
        m = numpy.ones_like(x) * dx * dx
        h = numpy.ones_like(x) * hdx * dx
        rho = numpy.ones_like(x)

        # create the particle array
        kernel = CubicSpline(dim=2)
        self.wdeltap = kernel.kernel(rij=dx, h=self.h)
        pa = get_particle_array_elastic_dynamics(
            name="solid", x=x + spacing, y=y, m=m,
            rho=rho, h=h, constants=dict(
                wdeltap=self.wdeltap, n=4, rho_ref=self.rho0,
                E=self.E, nu=self.nu))

        print('Ellastic Collision with %d particles' % (x.size))
        print("Shear modulus G = %g, Young's modulus = %g, Poisson's ratio =%g"
              % (pa.G, pa.E, pa.nu))

        u_f = 0.059
        pa.u = pa.cs * u_f * (2 * (x < 0) - 1)

        return [pa]

    def create_scheme(self):
        s = ElasticSolidsScheme(elastic_solids=['solid'], solids=[],
                                dim=2)
        s.configure_solver(dt=self.dt, tf=self.tf, pfreq=500)
        return s


if __name__ == '__main__':
    app = Rings()
    app.run()
