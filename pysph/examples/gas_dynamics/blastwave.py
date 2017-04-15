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
xmin = -0.5
xmax = 0.5


class Blastwave(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = 0.0
        self.rhol = 1.0
        self.rhor = 1.0
        self.pl = 1000.0
        self.pr = 0.01
        self.ul = 0.0
        self.ur = 0.0

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.5,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=200,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor/self.rhol
        self.nr = ratio*self.nl
        self.dxl = 0.5/self.nl
        self.dxr = 0.5/self.nr
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        return self.generate_particles(xmin=-0.5, xmax=0.5, dxl=self.dxl,
                                       dxr=self.dxr, m=self.dxl, pl=self.pl,
                                       pr=self.pr, h0=self.h0, bx=0.03,
                                       gamma1=gamma1)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=1.0, eps=0.5, g1=0.2, g2=0.4)

        # Running this will need to implement boundary condtion first.
        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s


if __name__ == '__main__':
    app = Blastwave()
    app.run()
    app.post_process()
