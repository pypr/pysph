"""2-d Riemann problem (xx minutes)

"""

# numpy and standard imports
import numpy

# pysph imports
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import \
    GSPHScheme, ADKEScheme, GasDScheme, SchemeChooser
from pysph.examples.gas_dynamics.riemann_2d_config import R2DConfig

# current case from the al possible unique cases
case = 3

# config for current case
config = R2DConfig(case)
gamma = 1.4
gamma1 = gamma - 1
kernel_factor = 1.5
dt = 1e-4
dim = 2

class Riemann2D(Application):
    def initialize(self):
        # square domain
        self.nx = 50
        self.ny = self.nx
        self.dt = dt
        self.tf = config.endtime

        self.dx = (config.xmax - config.xmin) / self.nx

    def create_particles_constant_volume(self):
        dx = self.dx
        dx2 = dx * 0.5
        vol = dx * dx

        xmin = config.xmin
        ymin = config.ymin
        xmax = config.xmax
        ymax = config.ymax
        xmid = config.xmid
        ymid = config.ymid

        rho1, u1, v1, p1 = config.rho1, config.u1, config.v1, config.p1
        rho2, u2, v2, p2 = config.rho2, config.u2, config.v2, config.p2
        rho3, u3, v3, p3 = config.rho3, config.u3, config.v3, config.p3
        rho4, u4, v4, p4 = config.rho4, config.u4, config.v4, config.p4
        x, y = numpy.mgrid[ xmin+dx2:xmax:dx,ymin+dx2:ymax:dx ]

        x = x.ravel(); y = y.ravel()

        u = numpy.zeros_like(x)
        v = numpy.zeros_like(x)

        # density and mass
        rho = numpy.ones_like(x)
        p = numpy.ones_like(x)

        for i in range(x.size):
            if x[i] <= xmid:
                if y[i] <= ymid: # w3
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                    v[i] = v3
                else:           # w2
                    rho[i] = rho2
                    p[i] = p2
                    u[i] = u2
                    v[i] = v2
            else:
                if y[i] <= ymid: # w4
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                    v[i] = v4
                else:           # w1
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                    v[i] = v1

        # thermal energy
        e = p/( gamma1 * rho )

        # mass
        m = vol * rho
        
        # smoothing length
        h = kernel_factor * (m/rho)**(1./dim)

        # create the particle array
        pa = gpa(name='fluid', x=x, y=y, m=m, rho=rho, h=h,
                 u=u, v=v, p=p, e=e)

        return pa

    def create_particles(self):
        fluid = self.create_particles_constant_volume()
        self.scheme.setup_properties([fluid])

        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=config.xmin, xmax=config.xmax,
            ymin=config.ymin, ymax=config.ymax,
            periodic_in_x=True, periodic_in_y=True
        )

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.25, g2=0.5, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6
        )

        s = SchemeChooser(
            default='gsph', gsph=gsph
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

if __name__ == "__main__":
    app = Riemann2D()
    app.run()