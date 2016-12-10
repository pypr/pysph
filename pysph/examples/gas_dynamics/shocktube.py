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

from pysph.sph.scheme import GasDScheme

# PySPH tools
from pysph.tools import uniform_distribution as ud

# Numerical constants
dim = 2
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 7.5e-6
tf = 0.0075

# domain size
xmin = -0.75; xmax = 0.75
dx = 0.01
ymin = 0; ymax = 30*dx

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.2
h0 = kernel_factor*dx


class ShockTube2D(Application):
    def create_domain(self):
        return DomainManager(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            periodic_in_x=True, periodic_in_y=True)

    def create_particles(self):
        global dx
        data = ud.uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax)

        x = data[0]; y = data[1]
        dx = data[2]; dy = data[3]

        # volume estimate
        volume = dx*dy

        # indices on either side of the initial discontinuity
        right_indices = numpy.where( x > 0.0 )[0]

        # density is uniform
        rho = numpy.ones_like(x)

        # pl = 100.0, pr = 0.1
        p = numpy.ones_like(x) * 1000.0
        p[right_indices] = 0.01

        # const h and mass
        h = numpy.ones_like(x) * h0
        m = numpy.ones_like(x) * volume * rho

        # thermal energy from the ideal gas EOS
        e = p/(gamma1*rho)

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m,
                    h0=h.copy())
        self.scheme.setup_properties([fluid])

        print("2D Shocktube with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,]

    def create_scheme(self):
        s = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta
        )
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False, pfreq=50)
        return s

if __name__ == '__main__':
    app = ShockTube2D()
    app.run()
