"""Classic Sod's shock-tube test. (5 seconds)

Two regions of a quiescient gas are separated by an imaginary
diaphgram that is instantaneously ruptured at t = 0. The two states
(left,right) are defined by the properties:

     left                               right

     density = 1.0                      density = 0.125
     pressure = 1.0                     pressure = 0.1

The solution examined at the final time T = 0.15s

"""

import os
# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application

from pysph.sph.scheme import ADKEScheme


# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 3e-4
tf = 0.15

# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
nl = 320; nr = 40
dxl = 0.5/nl; dxr = 8*dxl

# scheme constants
alpha = 1.0
beta = 1.0
kernel_factor = 1.2
h0 = kernel_factor*dxr


class ShockTube(Application):

    def generate_particles(self, xmin, xmax, dxl, dxr, m, pl, pr, h0,  bx,
            constants={}):
        xt1 = numpy.arange(xmin-bx + 0.5*dxl, 0, dxl)
        xt2 = numpy.arange(0.5*dxr, xmax+bx, dxr )
        xt = numpy.concatenate( [xt1, xt2] )
        leftb_indices = numpy.where( xt <= xmin)[0]
        left_indices =  numpy.where( (xt > xmin) & (xt < 0 ))[0]
        right_indices = numpy.where( (xt >=  0) & (xt < xmax))[0]
        rightb_indices = numpy.where( xt >= xmax)[0]
        x1 = xt[left_indices]
        print(x1)
        x2 = xt[right_indices]
        print(x2)
        b1 = xt[leftb_indices]
        print(b1)
        b2 = xt[rightb_indices]
        print(b2)

        x = numpy.concatenate( [x1, x2] )
        b = numpy.concatenate( [b1, b2] )
        right_indices = numpy.where(x>0.0)[0]

        rho = numpy.ones_like(x)*m/dxl
        rho[right_indices] = m/dxr

        p = numpy.ones_like(x)*pl
        p[right_indices] = pr

        h = numpy.ones_like(x) * h0
        m = numpy.ones_like(x)*m
        e = p/(gamma1*rho)
        wij = numpy.ones_like(x)

        bwij = numpy.ones_like(b)
        brho = numpy.ones_like(b)
        bp = numpy.ones_like(b)
        be = bp/(gamma1*brho)
        bm = numpy.ones_like(b)*dxl
        bh = numpy.ones_like(b) *h0
        bhtmp = numpy.ones_like(b)
        fluid = gpa(
                constants=constants, name='fluid', x=x, rho=rho, p=p,
                e=e, h=h, m=m, wij=wij, h0=h.copy()
                )

        boundary = gpa(
                constants=constants, name='boundary', x=b, rho=brho, p=bp,
                e=be, h=bh, m=bm, wij=bwij, h0=bh.copy(), htmp =bhtmp
                )

        self.scheme.setup_properties([fluid, boundary])
        print("1D Shocktube with %d particles"%(fluid.get_number_of_particles()))

        return [fluid, boundary]




    def create_particles(self):
        lng = numpy.zeros(1, dtype=float)
        consts ={ 'lng': lng}

        return self.generate_particles(xmin = -0.5, xmax=0.5, dxl=dxl, dxr=dxr,
                m=dxl, pl=1.0, pr=0.1, h0=h0, bx=0.03, constants=consts)

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
        x = pa.x
        rho = pa.rho
        e = pa.e
        cs = pa.cs
        u = pa.u
        plt.plot(x, rho)
        plt.xlabel('x'); plt.ylabel('rho')
        fig = os.path.join(self.output_dir, "density.png")
        plt.savefig(fig, dpi=300)
        plt.clf()
        plt.plot(x, e)
        plt.xlabel('x'); plt.ylabel('e')
        fig = os.path.join(self.output_dir, "energy.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        plt.plot(x,u/cs)
        plt.xlabel('x'); plt.ylabel('M')
        fig = os.path.join(self.output_dir, "Machno.png")
        plt.savefig(fig, dpi=300)
        plt.clf()



if __name__ == '__main__':
    app = ShockTube()
    app.run()
    app.post_process()
