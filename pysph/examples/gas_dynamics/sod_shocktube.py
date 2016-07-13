"""Simulate the classical Sod Shocktube problem in 1D (5 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, SchemeChooser
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 3e-4
tf = 0.15

# domain size and discretization parameters

nl = 320; nr = 40;
dxl = 0.5/nl; dxr = 0.5/nr

kernel_factor = 1.2
h0 = kernel_factor*dxr


class SodShockTube(ShockTubeSetup):
    def create_particles(self):
        return self.generate_particles(xmin=-0.5, xmax=0.5, dxl=dxl, dxr=dxr,
                m=dxl, pl=1.0, pr=0.1, h0=h0, bx=0.03, gamma1=gamma1,
                ul=0.0, ur=0.0)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=1,
            beta=1.0, k=0.3, eps=0.5, g1=0.2, g2=0.4)


        mpm = GasDScheme(
                fluids=['fluid'], solids = [], dim=dim, gamma=gamma,
                kernel_factor=kernel_factor, alpha1=1.0, alpha2=0.1,
                beta=2.0, update_alpha1=True, update_alpha2=True
                )
        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s




if __name__ == '__main__':
    app = SodShockTube()
    app.run()
    app.post_process()
