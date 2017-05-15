"""Simulate the Robert's problem (1D) (40 seconds).
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
xmin = -0.5
xmax = 0.5


class Robert(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = 0.0
        self.rhol = 3.86
        self.rhor = 1.0
        self.pl = 10.33
        self.pr = 1.0
        self.ul = -0.39
        self.ur = -3.02

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=2.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=1930,
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
        return self.generate_particles(xmin=-0.5, xmax=1, dxl=self.dxl,
                                       dxr=self.dxr, m=self.dxr, pl=self.pl,
                                       pr=self.pr, h0=self.h0, bx=0.03,
                                       gamma1=gamma1, ul=self.ul, ur=self.ur)

    def create_scheme(self):
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False, pfreq=50)
        return s

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            alpha=1, beta=2.0, k=1.0, eps=0.5, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=2.0, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )
        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s


if __name__ == '__main__':
    app = Robert()
    app.run()
    app.post_process()
