import os
import numpy
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, SchemeChooser


# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-6
tf = 0.0075

# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
nl = 200; nr = 200
dxl = 0.5/nl; dxr = dxl

kernel_factor = 1.5
h0 = kernel_factor*dxr


class Blastwave(ShockTubeSetup):

    def create_particles(self):
        lng = numpy.zeros(1, dtype=float)
        consts ={ 'lng': lng}

        return self.generate_particles(xmin = -0.5, xmax=0.5, dxl=dxl, dxr=dxr,
                m=dxl, pl=1000.0, pr=0.01, h0=h0, bx=0.03,
                gamma1=gamma1, constants=consts)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=1,
            beta=1, k=1.0, eps=0.5, g1=0.2, g2=0.4)

        s = SchemeChooser(default='adke', adke=adke)
        return s



if __name__ == '__main__':
    app = Blastwave()
    app.run()
    app.post_process()
