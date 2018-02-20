"""Two square blocks of water colliding with each other. (20 seconds)
"""

import numpy
from pysph.examples._db_geometry import create_2D_filled_region
from pysph.base.utils import get_particle_array
from pysph.sph.iisph import IISPHScheme
from pysph.solver.application import Application

dx = 0.025
hdx = 1.0
rho0 = 1000


class TwoBlocks(Application):
    def create_particles(self):
        x1, y1 = create_2D_filled_region(-1, 0, 0, 1, dx)
        x2, y2 = create_2D_filled_region(0.5, 0, 1.5, 1, dx)

        x = numpy.concatenate((x1, x2))
        y = numpy.concatenate((y1, y2))
        u1 = numpy.ones_like(x1)
        u2 = -numpy.ones_like(x2)
        u = numpy.concatenate((u1, u2))

        rho = numpy.ones_like(u)*rho0
        h = numpy.ones_like(u)*hdx*dx
        m = numpy.ones_like(u)*dx*dx*rho0

        fluid = get_particle_array(
            name='fluid', x=x, y=y, u=u, rho=rho, m=m, h=h
        )
        self.scheme.setup_properties([fluid])
        return [fluid]

    def create_scheme(self):
        s = IISPHScheme(fluids=['fluid'], solids=[], dim=2, rho0=rho0)
        return s

    def configure_scheme(self):
        dt = 2e-3
        tf = 1.0
        self.scheme.configure_solver(
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=10
        )


if __name__ == '__main__':
    app = TwoBlocks()
    app.run()
