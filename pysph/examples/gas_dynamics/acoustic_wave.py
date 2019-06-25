r"""Diffusion of an acoustic wave in 1-d (5 minutes)

Propagation of acoustic wave
particles have properties according
to the following distribuion
.. math::
        \rho = \rho_0 + \Delta\rho sin(kx)
        p = p_0 + c_0^2\Delta\rho sin(kx)
        u = c_0\rho_0^{-1}\Delta\rho sin(kx)

with :math:`\Delta\rho = 1e-6` and :math:`k = 2\pi/\lambda`
where :math:`\lambda` is the domain length.
.. math::
        \rho_0 = \gamma = 1.4 and p_0 = 1.0
"""


# standard library and numpy imports
import numpy

# pysph imports
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import \
    GSPHScheme, ADKEScheme, GasDScheme, SchemeChooser
from pysph.sph.wc.crksph import CRKSPHScheme


class AcousticWave(Application):
    def initialize(self):
        self.xmin = 0.
        self.xmax = 1.
        self.gamma = 1.4
        self.rho_0 = self.gamma
        self.p_0 = 1.
        self.c_0 = 1.
        self.delta_rho = 1e-6
        self.n_particles = 8
        self.domain_length = self.xmax - self.xmin
        self.k = -2 * numpy.pi / self.domain_length
        self.cfl = 0.1
        self.hdx = 1.0
        self.dt = 1e-3
        self.tf = 5
        self.dim = 1

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=1, periodic_in_x=True
        )

    def add_user_options(self, group):
        group.add_argument(
            "--nparticles", action="store", type=int, dest="nprt", default=256,
            help="Number of particles in domain"
        )

    def consume_user_options(self):
        self.n_particles = self.options.nprt
        self.dx = self.domain_length / (self.n_particles)
        self.dt = self.cfl * self.dx / self.c_0

    def create_particles(self):
        x = numpy.arange(
            self.xmin + self.dx*0.5, self.xmax, self.dx
        )
        rho = self.rho_0 + self.delta_rho *\
            numpy.sin(self.k * x)

        p = self.p_0 + self.c_0**2 *\
            self.delta_rho * numpy.sin(self.k * x)

        u = self.c_0 * self.delta_rho * numpy.sin(self.k * x) /\
            self.rho_0
        cs = numpy.sqrt(
            self.gamma * p / rho
        )
        h = numpy.ones_like(x) * self.dx * self.hdx
        m = numpy.ones_like(x) * self.dx * rho
        e = p / ((self.gamma - 1) * rho)
        fluid = gpa(
            name='fluid', x=x, p=p, rho=rho, u=u, h=h, m=m, e=e, cs=cs,
            h0=h.copy()
        )
        self.scheme.setup_properties([fluid])

        return [fluid, ]

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=self.dim,
            gamma=self.gamma, kernel_factor=1.0,
            g1=0., g2=0., rsolver=7, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=5.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=self.dim, gamma=self.gamma,
            kernel_factor=1.2, alpha1=0, alpha2=0,
            beta=2.0, update_alpha1=False, update_alpha2=False,
            has_ghosts=True
        )

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=self.dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=self.gamma, cl=2, has_ghosts=True
        )

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=self.dim, gamma=self.gamma,
            alpha=0, beta=0.0, k=1.5, eps=0.0, g1=0.0, g2=0.0,
            has_ghosts=True)

        s = SchemeChooser(
            default='gsph', gsph=gsph, mpm=mpm, crksph=crksph, adke=adke
        )

        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(
                dt=self.dt, tf=self.tf,
                adaptive_timestep=True, pfreq=50
            )

        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

        if self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

        if self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

    def post_process(self):
        from pysph.solver.utils import load
        if len(self.output_files) < 1:
            return
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']
        x_c = pa.x
        u = self.c_0 * self.delta_rho * numpy.sin(self.k * x_c) /\
            self.rho_0
        u_c = pa.u
        l_inf = numpy.max(
            numpy.abs(u_c - u)
        )
        l_1 = (numpy.sum(
            numpy.abs(u_c - u)
        ) / self.n_particles)
        print("L_inf norm of velocity for the problem: %s" % (l_inf))
        print("L_1 norm of velocity for the problem: %s" % (l_1))

        rho = self.rho_0 + self.delta_rho *\
            numpy.sin(self.k * x_c)

        rho_c = pa.rho
        l1 = numpy.sum(
            numpy.abs(rho - rho_c)
        )
        l1 = l1 / self.n_particles
        print("l_1 norm of density for the problem: %s" % (l1))


if __name__ == "__main__":
    app = AcousticWave()
    app.run()
    app.post_process()
