import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme


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
        self.scheme.setup_properties([pa])
        return [pa]

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=1.0, c0=1400,
            h0=1.3*0.025, hdx=1.3, gamma=7.0, alpha=0.1, beta=0.0
        )
        dt = 5e-6
        tf = 0.0076
        s.configure_solver(
            dt=dt, tf=tf,
        )
        return s


if __name__ == '__main__':
    app = EllipticalDrop(fname='ed')
    app.run()
