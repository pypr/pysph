""" The Kelvin-Helmholtz instability test (1.5 hours)
"""

# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme, SchemeChooser, ADKEScheme, GSPHScheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.base.nnps import DomainManager
from pysph.tools import uniform_distribution as ud


# problem constants
dim = 2
gamma = 5.0/3.0

xmin = ymin = 0.0
xmax = ymax = 1.0

rhoi_1 = 1
rhoi_2 = 2
rhoi_m = 0.5 * (rhoi_1 - rhoi_2)

v_i1 = 0.5
v_i2 = -0.5
v_im = 0.5 * (v_i1 - v_i2)

delta = 0.025
dely = 0.01
wavelen = 0.5

dt = 1e-3

tf = 2


class KHInstability(Application):
    def initialize(self):
        self.nx = 200
        self.dx = (xmax - xmin) / self.nx
        self.dy = self.dx
        self.hdx = 1.5

    def create_particles(self):
        data = ud.uniform_distribution_cubic2D(
            self.dx, xmin, xmax, ymin, ymax
        )

        x = data[0].ravel()
        y = data[1].ravel()

        y1 = numpy.where(
            (y >= 0) & (y < 0.25)
        )[0]

        y2 = numpy.where(
            (y >= 0.25) & (y < 0.5)
        )[0]

        y3 = numpy.where(
            (y >= 0.5) & (y < 0.75)
        )[0]

        y4 = numpy.where(
            (y >= 0.75) & (y < 1.0)
        )[0]

        rho1 = rhoi_1 - rhoi_m * numpy.exp((y[y1] - 0.25)/delta)
        rho2 = rhoi_2 + rhoi_m * numpy.exp((0.25 - y[y2])/delta)
        rho3 = rhoi_2 + rhoi_m * numpy.exp((y[y3] - 0.75)/delta)
        rho4 = rhoi_1 - rhoi_m * numpy.exp((0.75 - y[y4])/delta)

        u1 = v_i1 - v_im * numpy.exp((y[y1] - 0.25)/delta)
        u2 = v_i2 + v_im * numpy.exp((0.25 - y[y2])/delta)
        u3 = v_i2 + v_im * numpy.exp((y[y3] - 0.75)/delta)
        u4 = v_i1 - v_im * numpy.exp((0.75 - y[y4])/delta)

        v = dely * numpy.sin(
            2 * numpy.pi * x / wavelen
        )

        p = 2.5

        rho = numpy.concatenate((
            rho1, rho2, rho3, rho4
        ))

        u = numpy.concatenate((
            u1, u2, u3, u4
        ))

        v = numpy.concatenate((
            v[y1], v[y2], v[y3], v[y4]
        ))

        x = numpy.concatenate((
            x[y1], x[y2], x[y3], x[y4]
        ))

        y = numpy.concatenate((
            y[y1], y[y2], y[y3], y[y4]
        ))

        e = p / ((gamma - 1) * rho)

        m = self.dx * self.dx * rho

        h = self.dx * self.hdx

        fluid = gpa(
            name='fluid', x=x, y=y, u=u, v=v, rho=rho, p=p, e=e, m=m, h=h,
            h0=h
        )

        self.scheme.setup_properties([fluid])
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            periodic_in_x=True, periodic_in_y=True
        )

    def create_scheme(self):
        crk = CRKSPHScheme(
            fluids=['fluid'], dim=2, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=2, has_ghosts=True
        )

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=0.1, beta=0.1, k=1.2, eps=0.1, g1=0.1, g2=0.2,
            has_ghosts=True)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
            has_ghosts=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        s = SchemeChooser(
            default='crksph', crksph=crk, gsph=gsph, adke=adke, mpm=mpm
        )

        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'crksph':
            s.configure_solver(
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=50
            )
        elif self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(dt=dt, tf=tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=dt, tf=tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=dt, tf=tf,
                               adaptive_timestep=False, pfreq=50)


if __name__ == "__main__":
    app = KHInstability()
    app.run()
