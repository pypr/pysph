import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application


class EllipticalDrop(Application):
    def create_particles(self):
        dx = 0.025
        x, y = np.mgrid[-1.05:1.05:dx, -1.05:1.05:dx]
        mask = x*x + y*y < 1
        x = x[mask]
        y = y[mask]
        rho = 1.0
        h = 1.3*dx
        m = rho*dx*dx
        pa = get_particle_array(
            name='fluid', x=x, y=y, u=-100*x, v=100*y, rho=rho,
            m=m, h=h
        )
        return [pa]


if __name__ == '__main__':
    app = EllipticalDrop(fname='ed')
    app.run()
