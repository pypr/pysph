"""Evolution of a circular patch of incompressible fluid. (30 seconds)

This is the simplest implementation using existing schemes.

See J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994, 100, pp
399 - 406
"""
from __future__ import print_function

from numpy import ones_like, mgrid, sqrt

from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme


class EllipticalDrop(Application):
    def initialize(self):
        self.co = 1400.0
        self.ro = 1.0
        self.hdx = 1.3
        self.dx = 0.025
        self.alpha = 0.1

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=self.ro, c0=self.co,
            h0=self.dx*self.hdx, hdx=self.hdx, gamma=7.0, alpha=0.1, beta=0.0
        )
        dt = 5e-6
        tf = 0.0076
        s.configure_solver(dt=dt, tf=tf)
        return s

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        ro = self.ro
        name = 'fluid'
        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x)*dx*dx*ro
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) * ro
        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
                                name=name)
        pa.remove_particles(indices)

        print("Elliptical drop :: %d particles"
              % (pa.get_number_of_particles()))

        self.scheme.setup_properties([pa])
        return [pa]


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
