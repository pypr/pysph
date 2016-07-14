"""Simulate a 1D blast wave problem (30 seconds).
"""
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
        return self.generate_particles(xmin = -0.5, xmax=0.5, dxl=dxl, dxr=dxr,
                m=dxl, pl=1000.0, pr=0.01, h0=h0, bx=0.03,
                gamma1=gamma1)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
                fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=1,
                beta=1, k=1.0, eps=0.5, g1=0.2, g2=0.4)

        # Running this will need to implement boundary condtion first.
        mpm = GasDScheme(
                fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
                kernel_factor=kernel_factor, alpha1=1.0, alpha2=0.1,
                beta=2.0, update_alpha1=True, update_alpha2=True
                )

        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s



if __name__ == '__main__':
    app = Blastwave()
    app.run()
    app.post_process()
