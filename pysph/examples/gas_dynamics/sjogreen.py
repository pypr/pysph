"""Simulate the Sjogreen problem in 1D (10 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, SchemeChooser
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
nl = 200.; nr = 200.0
dxl = 0.5/nl; dxr = 0.5/nr

kernel_factor = 2.5
h0 = kernel_factor*dxr


class SjoGreen(ShockTubeSetup):
    def create_particles(self):
        lng = numpy.zeros(1, dtype=float)
        consts ={ 'lng': lng}

        return self.generate_particles(xmin = -0.5, xmax=0.5, dxl=dxl, dxr=dxr,
                m=dxl, pl=0.4, pr=0.4, h0=h0, bx=0.03, gamma1=gamma1,
                ul=-2, ur=2, constants=consts)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=0,
            beta=0.0, k=1.0, eps=1.0, g1=0.0, g2=0.0)

        mpm = GasDScheme(
                fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
                kernel_factor=kernel_factor, alpha1=0, alpha2=0,
                beta=2.0, update_alpha1=True, update_alpha2=True
                )
        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s

if __name__ == '__main__':
    app = SjoGreen()
    app.run()
    app.post_process()
