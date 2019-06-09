r"""Cheng and Shu's 1d acoustic wave propagation in 1d (1 min)

particles have properties according
to the following distribuion
.. math::
        \rho = \rho_0 + \Delta\rho sin(kx)
        p = 1.0
        u = 1 + 0.1sin(kx)

with :math:`\Delta\rho = 1` and :math:`k = 2\pi/\lambda`
where \lambda is the domain length.
.. math::
        \rho_0 = 2, \gamma = 1.4 and p_0 = 1.0
"""


# standard library and numpy imports
import numpy

# pysph imports
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import GSPHScheme, SchemeChooser


class ChengShu(Application):
    def initialize(self):
        self.xmin = 0.
        self.xmax = 1.
        self.gamma = 1.4
        self.p_0 = 1.
        self.c_0 = 1.
        self.delta_rho = 1
        self.n_particles = 1000
        self.domain_length = self.xmax - self.xmin
        self.dx = self.domain_length / (self.n_particles - 1)
        self.k = 2 * numpy.pi / self.domain_length
        self.hdx = 2.
        self.dt = 1e-4
        self.tf = 1.0
        self.dim = 1

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, periodic_in_x=True
        )

    def create_particles(self):
        x = numpy.linspace(
            self.xmin, self.xmax, self.n_particles
            )
        rho = 2 + numpy.sin(2 * numpy.pi * x)*self.delta_rho

        p = numpy.ones_like(x)

        u = 1 + 0.1 * numpy.sin(2 * numpy.pi * x)

        cs = numpy.sqrt(
            self.gamma * p / rho
        )
        h = numpy.ones_like(x) * self.dx * self.hdx
        m = numpy.ones_like(x) * self.dx * rho
        e = p / ((self.gamma - 1) * rho)

        fluid = gpa(
            name='fluid', x=x, p=p, rho=rho, u=u, h=h, m=m, e=e, cs=cs
        )

        self.scheme.setup_properties([fluid])

        return [fluid, ]

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=self.dim,
            gamma=self.gamma, kernel_factor=1.,
            g1=0., g2=0., rsolver=3, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=5.0,
            niter=200, tol=1e-6
        )

        s = SchemeChooser(
            default='gsph', gsph=gsph
        )

        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(
                dt=self.dt, tf=self.tf,
                adaptive_timestep=False, pfreq=1000
            )


if __name__ == "__main__":
    app = ChengShu()
    app.run()
