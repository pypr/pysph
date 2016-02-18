"""Example for the Noh's cylindrical implosion test. (15 mins)
"""

# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme


# problem constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.2

# numerical constants
dt = 1e-4
tf = 0.4

# domain and particle spacings
xmin = ymin = -0.5
xmax = ymax = 0.5

nx = ny = 50
dx = (xmax-xmin)/nx
dxb2 = 0.5 * dx

# initial values
h0 = kernel_factor*dx
rho0 = 1.0
e0 = 1e-6
m0 = dx*dx * rho0
vr = -1.0


class NohImplosion(Application):
    def create_particles(self):
        x, y = numpy.mgrid[
            xmin:xmax:dx, ymin:ymax:dx]

        # positions
        x = x.ravel(); y = y.ravel()

        rho = numpy.ones_like(x) * rho0
        m = numpy.ones_like(x) * m0
        e = numpy.ones_like(x) * e0
        h = numpy.ones_like(x) * h0
        p = gamma1*rho*e

        u = numpy.ones_like(x)
        v = numpy.ones_like(x)

        sin, cos, arctan = numpy.sin, numpy.cos, numpy.arctan2
        for i in range(x.size):
            theta = arctan(y[i],x[i])
            u[i] = vr*cos(theta)
            v[i] = vr*sin(theta)

        fluid = gpa(name='fluid', x=x,y=y,m=m,rho=rho, h=h,u=u,v=v,p=p,e=e)
        self.scheme.setup_properties([fluid])

        print("Noh's problem with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,]

    def create_scheme(self):
        s = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, adaptive_h_scheme="gsph",
            update_alpha1=True, update_alpha2=True
        )
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False)
        return s


if __name__ == '__main__':
    app = NohImplosion()
    app.run()
