"""Two-dimensional Shocktube problem. (10 mins)

The density is assumed to be uniform and the shocktube problem is
defined by the pressure jump. The pressure jump of 10^5 (pl = 1000.0,
pr = 0.01) corresponds to the Woodward and Colella strong shock or
blastwave problem.

"""

# NumPy and standard library imports
import numpy

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

from pysph.sph.scheme import GasDScheme, ADKEScheme, GSPHScheme, SchemeChooser
from pysph.sph.wc.crksph import CRKSPHScheme

# PySPH tools
from pysph.tools import uniform_distribution as ud

# Numerical constants
dim = 2
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 7.5e-6
tf = 0.005


# domain size
xmin = 0.
xmax = 1
dx = 0.002
ny = 50
ymin = 0
ymax = ny * dx
x0 = 0.5  # initial discontuinity

# scheme constants
alpha1 = 1.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.5
h0 = kernel_factor * dx


class ShockTube2D(Application):
    def initialize(self):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = dx
        self.hdx = 1.7
        self.x0 = x0
        self.ny = ny
        self.pl = 1000
        self.pr = 0.01
        self.rhol = 1.0
        self.rhor = 1.0
        self.ul = 0.
        self.ur = 0.
        self.vl = 0.
        self.vr = 0.

    def create_domain(self):
        return DomainManager(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            periodic_in_x=True, periodic_in_y=True)

    def create_particles(self):
        global dx
        data = ud.uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax)

        x = data[0]
        y = data[1]
        dx = data[2]
        dy = data[3]

        # volume estimate
        volume = dx * dy

        # indices on either side of the initial discontinuity
        right_indices = numpy.where(x > x0)[0]

        # density is uniform
        rho = numpy.ones_like(x) * self.rhol
        rho[right_indices] = self.rhor

        # pl = 100.0, pr = 0.1
        p = numpy.ones_like(x) * self.pl
        p[right_indices] = self.pr

        # const h and mass
        h = numpy.ones_like(x) * self.hdx * self.dx
        m = numpy.ones_like(x) * volume * rho

        # ul = ur = 0
        u = numpy.ones_like(x) * self.ul
        u[right_indices] = self.ur

        # vl = vr = 0
        v = numpy.ones_like(x) * self.vl
        v[right_indices] = self.vr

        # thermal energy from the ideal gas EOS
        e = p/(gamma1*rho)

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m,
                    h0=h.copy(), u=u, v=v)
        self.scheme.setup_properties([fluid])

        print("2D Shocktube with %d particles" %
              (fluid.get_number_of_particles()))

        return [fluid, ]

    def create_scheme(self):
        self.dt = dt
        self.tf = tf

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=1.0, eps=0.8, g1=0.5, g2=0.5,
            has_ghosts=True)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, max_density_iterations=1000,
            density_iteration_tolerance=1e-4, has_ghosts=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.25, g2=0.5, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=2, has_ghosts=True
        )

        s = SchemeChooser(
            default='adke', adke=adke, mpm=mpm, gsph=gsph, crksph=crksph
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

    def post_process(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot
        except ImportError:
            print("Post processing requires matplotlib.")
            return

        if self.rank > 0 or len(self.output_files) == 0:
            return

        import os
        from pysph.solver.utils import load
        from pysph.examples.gas_dynamics import riemann_solver
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']

        try:
            gamma = self.options.gamma or 1.4
        except AttributeError:
            gamma = 1.4
        print(gamma)
        riemann_solver.set_gamma(gamma)
        rho_e, u_e, p_e, e_e, x_e = riemann_solver.solve(
            x_min=0, x_max=1, x_0=0.5,
            t=self.tf, p_l=self.pl, p_r=self.pr, rho_l=self.rhol,
            rho_r=self.rhor, u_l=self.ul, u_r=self.ur, N=101
        )

        x = pa.x
        u = pa.u
        e = pa.e
        p = pa.p
        rho = pa.rho
        cs = pa.cs

        pyplot.scatter(
            x, rho, label='pysph (' + str(self.options.scheme) + ')',
            s=1, color='k'
            )
        pyplot.plot(x_e, rho_e, label='exact')
        pyplot.xlim((0.2, 0.8))
        pyplot.xlabel('x')
        pyplot.ylabel('rho')
        pyplot.legend()
        fig = os.path.join(self.output_dir, "density.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.clf()

        pyplot.scatter(
            x, e, label='pysph (' + str(self.options.scheme) + ')',
            s=1, color='k'
            )
        pyplot.plot(x_e, e_e, label='exact')
        pyplot.xlim((0.2, 0.8))
        pyplot.xlabel('x')
        pyplot.ylabel('e')
        pyplot.legend()
        fig = os.path.join(self.output_dir, "energy.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.clf()

        pyplot.scatter(
            x, rho * u, label='pysph (' + str(self.options.scheme) + ')',
            s=1, color='k'
            )
        pyplot.plot(x_e, rho_e * u_e, label='exact')
        pyplot.xlim((0.2, 0.8))
        pyplot.xlabel('x')
        pyplot.ylabel('M')
        pyplot.legend()
        fig = os.path.join(self.output_dir, "Machno.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.clf()

        pyplot.scatter(
            x, p, label='pysph (' + str(self.options.scheme) + ')',
            s=1, color='k'
            )
        pyplot.plot(x_e, p_e, label='exact')
        pyplot.xlim((0.2, 0.8))
        pyplot.xlabel('x')
        pyplot.ylabel('p')
        pyplot.legend()
        fig = os.path.join(self.output_dir, "pressure.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.clf()

        fname = os.path.join(self.output_dir, 'results.npz')
        numpy.savez(fname, x=x, u=u, e=e, cs=cs, rho=rho, p=p)

        fname = os.path.join(self.output_dir, 'exact.npz')
        numpy.savez(fname, x=x_e, u=u_e, e=e_e, rho=rho_e, p=p_e)


if __name__ == '__main__':
    app = ShockTube2D()
    app.run()
    app.post_process()
