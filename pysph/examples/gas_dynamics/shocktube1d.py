"""Classic Sod's shock-tube test. (5 seconds)

Two regions of a quiescient gas are separated by an imaginary
diaphgram that is instantaneously ruptured at t = 0. The two states
(left,right) are defined by the properties:

     left                               right

     density = 1.0                      density = 0.125
     pressure = 1.0                     pressure = 0.1

The solution examined at the final time T = 0.15s

"""

# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

from pysph.sph.scheme import GasDScheme


# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.15

# domain size and discretization parameters
xmin = 0; xmax = 1.0
nl = 320; nr = 40
dxl = 0.5/nl; dxr = 8*dxl

# scheme constants
a1 = 1.0
a2 = 0.5
beta = 2.0
kernel_factor = 1.2
h0 = kernel_factor*dxr


class ShockTube1D(Application):
    def create_domain(self):
        return DomainManager(xmin=xmin, xmax=xmax, periodic_in_x=False)

    def create_particles(self):
        # particle positions
        x1 = numpy.arange( 0.5*dxl, 0.5, dxl )
        x2 = numpy.arange( 0.5 + 0.5*dxr, 1.0, dxr )
        x = numpy.concatenate( [x1, x2] )

        # indices on either side of the initial discontinuity
        right_indices = numpy.where( x > 0.5 )[0]

        # density
        rho = numpy.ones_like(x)
        rho[right_indices] = 0.125

        # pl = 1.0, pr = 0.1
        p = numpy.ones_like(x)
        p[right_indices] = 0.1

        # const h and mass
        h = numpy.ones_like(x) * h0
        m = numpy.ones_like(x) * dxl

        # thermal energy from the ideal gas EOS
        e = p/(gamma1*rho)

        # viscosity parameters
        alpha1 = numpy.ones_like(x) * a1
        alpha2 = numpy.ones_like(x) * a2

        fluid = gpa(name='fluid', x=x, rho=rho, p=p, e=e, h=h, m=m, h0=h.copy(),
                    alpha1=alpha1, alpha2=alpha2)

        self.scheme.setup_properties([fluid])
        print("1D Shocktube with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,]

    def create_scheme(self):
        s = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=a1, alpha2=a2,
            beta=beta, update_alpha1=True, update_alpha2=True
        )
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=True, pfreq=50)
        return s


if __name__ == '__main__':
    app = ShockTube1D()
    app.run()
