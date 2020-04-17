""" This simulates two square blocks of water colliding with each other.

This example solves exactly the same problem as the two_blocks.py but shows
how they can be treated as different fluids.

"""

import numpy
from pysph.examples._db_geometry import create_2D_filled_region
from pysph.base.utils import get_particle_array
from pysph.sph.iisph import IISPHScheme
from pysph.solver.application import Application

dx = 0.025
hdx = 1.0
rho0 = 1000


class TwoBlocks2Body(Application):
    def create_particles(self):
        x1, y1 = create_2D_filled_region(-1, 0, 0, 1, dx)
        x2, y2 = create_2D_filled_region(0.5, 0, 1.5, 1, dx)

        u1 = numpy.ones_like(x1)
        u2 = -numpy.ones_like(x2)

        rho = numpy.ones_like(x1)*rho0
        h = numpy.ones_like(u1)*hdx*dx
        m = numpy.ones_like(u1)*dx*dx*rho0

        fluid1 = get_particle_array(
            name='fluid1', x=x1, y=y1, u=u1, rho=rho, m=m, h=h
        )
        fluid2 = get_particle_array(
            name='fluid2', x=x2, y=y2, u=u2, rho=rho, m=m, h=h
        )
        self.scheme.setup_properties([fluid1, fluid2])
        return [fluid1, fluid2]

    def create_scheme(self):
        s = IISPHScheme(fluids=['fluid1', 'fluid2'], solids=[], dim=2,
                        rho0=rho0)
        return s

    def configure_scheme(self):
        dt = 2e-3
        tf = 1.0
        self.scheme.configure_solver(
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=10
        )


if __name__ == '__main__':
    app = TwoBlocks2Body()
    app.run()
