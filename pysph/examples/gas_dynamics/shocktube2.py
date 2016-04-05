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

from pysph.sph.gasd_schemes import ADKEScheme


# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 3e-4
tf = 0.15

# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
nl = 400; nr = 50
dxl = 0.5/nl; dxr = dxl/0.125

# scheme constants
alpha = 1.0
beta = 2.0
kernel_factor = 1.2
h0 = kernel_factor*dxr

bh0 = 10*dxr

class ShockTube2(Application):
    def create_domain(self):
        return DomainManager(xmin=xmin, xmax=xmax, periodic_in_x=False)

    def create_particles(self):
        # particle positions
        xt1 = numpy.arange( -0.60 + 0.5*dxl, 0, dxl )
        xt2 = numpy.arange(  0.5*dxr, 0.60 , dxr )

        xt = numpy.concatenate( [xt1, xt2] )
        leftb_indices = numpy.where( xt <= -0.5)[0]
        left_indices =  numpy.where( (xt > -0.5) & (xt < 0 ))[0]
        right_indices = numpy.where( (xt >=  0) & (xt < 0.5))[0]
        rightb_indices = numpy.where( xt >= 0.5)[0]
        x1 = xt[left_indices]
        x2 = xt[right_indices]
        b1 = xt[leftb_indices]
        b2 = xt[rightb_indices]

        x = numpy.concatenate( [x1, x2] )
        b = numpy.concatenate( [b1, b2] )

        right_indices = numpy.where(x>0.0)[0]
        rightb_indices = numpy.where( b >= 0.5)[0]
        # density
        rho = numpy.ones_like(x)
        rho[right_indices] = 0.125

        # pl = 1.0, pr = 0.1
        p = numpy.ones_like(x)
        p[right_indices] = 0.1

        # const h and mass
        h = numpy.ones_like(x) * h0
        h_0 = numpy.ones_like(x) * h0
        m = numpy.ones_like(x) * dxl

        # thermal energy from the ideal gas EOS
        e = p/(gamma1*rho)


        wij = numpy.ones_like(x)

        bwij = numpy.ones_like(b)
        brho = numpy.ones_like(b)
        brho[rightb_indices] = 0.125
        bp = numpy.ones_like(b)
        bp[rightb_indices] = 0.1
        be = bp/(gamma1*brho)
        bm = numpy.ones_like(b)*dxl
        bh = numpy.ones_like(b) * bh0
        bh_0 = numpy.ones_like(b) * bh0
        bhtmp = numpy.ones_like(b) * bh0
        lng = numpy.zeros(1, dtype=float)
        consts ={ 'lng': lng}
        fluid = gpa(
                constants=consts,name='fluid', x=x, rho=rho, p=p,
                e=e, h=h, m=m, wij=wij, h0=h_0
                )

        solid = gpa(
                constants=consts,name='boundary', x=b, rho=brho, p=bp,
                e=be, h=bh, m=bm, wij=bwij, h0=bh_0, htmp =bhtmp
                )

        self.scheme.setup_properties([fluid, solid])
        print("1D Shocktube with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,solid]

    def create_scheme(self):
        s = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=1,
            beta=1, k=0.3, eps=0.5, g1=0.2, g2=0.4)

        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False, pfreq=50)
        return s

    def post_process(self):
        from matplotlib import pyplot as plt
        last_output = self.output_files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        plt.plot(pa.x,pa.rho)
        plt.plot(pa.x,pa.e)
        plt.show()


if __name__ == '__main__':
    app = ShockTube2()
    app.run()
    app.post_process()
