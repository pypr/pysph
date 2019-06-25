"""Example for the Noh's cylindrical implosion test. (10 minutes)
"""

# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme, SchemeChooser, ADKEScheme, GSPHScheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.base.nnps import DomainManager

# problem constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.5

# numerical constants
dt = 1e-3
tf = 0.6

# domain and particle spacings
xmin = ymin = -1.0
xmax = ymax = 1.0

nx = ny = 100
dx = (xmax-xmin)/nx
dxb2 = 0.5 * dx

# initial values
h0 = kernel_factor*dx
rho0 = 1.0
m0 = dx*dx * rho0
vr = -1.0


class NohImplosion(Application):
    def create_particles(self):
        x, y = numpy.mgrid[
            xmin:xmax:dx, ymin:ymax:dx]

        # positions
        x = x.ravel()
        y = y.ravel()

        rho = numpy.ones_like(x) * rho0
        m = numpy.ones_like(x) * m0
        h = numpy.ones_like(x) * h0

        u = numpy.ones_like(x)
        v = numpy.ones_like(x)

        sin, cos, arctan = numpy.sin, numpy.cos, numpy.arctan2
        for i in range(x.size):
            theta = arctan(y[i], x[i])
            u[i] = vr*cos(theta)
            v[i] = vr*sin(theta)

        fluid = gpa(
            name='fluid', x=x, y=y, m=m, rho=rho, h=h, u=u, v=v, p=1e-12,
            e=2.5e-11, h0=h.copy()
            )
        self.scheme.setup_properties([fluid])

        print("Noh's problem with %d particles "
              % (fluid.get_number_of_particles()))

        return [fluid, ]

    def create_domain(self):
        return DomainManager(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            mirror_in_x=True, mirror_in_y=True
        )

    def create_scheme(self):
        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, adaptive_h_scheme="mpm",
            update_alpha1=True, update_alpha2=True, has_ghosts=True
        )

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=2, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=2, has_ghosts=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.25, g2=0.5, rsolver=7, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=1.0, eps=0.8, g1=0.5, g2=0.5,
            has_ghosts=True)

        s = SchemeChooser(
            default='crksph', crksph=crksph, mpm=mpm, adke=adke, gsph=gsph
        )
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False)
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(
                dt=dt, tf=tf, adaptive_timestep=True, pfreq=50
            )
        elif self.options.scheme == 'crksph':
            s.configure_solver(
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=50
            )
        elif self.options.scheme == 'gsph':
            s.configure_solver(
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=50
            )
        elif self.options.scheme == 'adke':
            s.configure_solver(
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=50
            )

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
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']

        x = pa.x
        y = pa.y
        rho = pa.rho
        p = pa.p

        r = numpy.sqrt(x**2 + y**2)

        # exact solutions
        vs = 1.0/3.0  # shock radial velocity
        rs = vs * tf  # position of shock
        ri = numpy.linspace(0, rs, 10)
        ro = numpy.linspace(rs, xmax, 100)
        re = numpy.concatenate((ri, ro))

        rho_e1 = numpy.ones_like(ri) * ((gamma + 1) / (gamma - 1))**dim
        rho_e2 = rho0 * (1 + tf / ro)**(dim - 1)
        rho_e = numpy.concatenate((rho_e1, rho_e2))

        p_e1 = vs * rho_e1
        p_e2 = numpy.zeros_like(ro)
        p_e = numpy.concatenate((p_e1, p_e2))

        pyplot.scatter(r, p, s=1)
        pyplot.xlabel('r')
        pyplot.ylabel('P')
        pyplot.plot(re, p_e, color='r', lw=1)
        pyplot.legend(
            ['exact', self.options.scheme]
        )
        fname = os.path.join(self.output_dir, 'pressure.png')
        pyplot.savefig(fname, dpi=300)
        pyplot.close('all')

        pyplot.scatter(r, rho, s=1)
        pyplot.xlabel('r')
        pyplot.ylabel(r'$\rho$')
        pyplot.plot(re, rho_e, color='r', lw=1)
        pyplot.legend(
            ['exact', self.options.scheme]
        )
        fname = os.path.join(self.output_dir, 'density.png')
        pyplot.savefig(fname, dpi=300)
        pyplot.close('all')


if __name__ == '__main__':
    app = NohImplosion()
    app.run()
    app.post_process()
