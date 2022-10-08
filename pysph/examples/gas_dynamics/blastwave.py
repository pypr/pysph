"""Simulate a 1D blast wave problem (30 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import (ADKEScheme, GasDScheme, GSPHScheme,
                              SchemeChooser, add_bool_argument)
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme

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
        add_bool_argument(group, 'smooth-ic', dest='smooth_ic', default=False,
                          help="Smooth the initial condition.")

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        self.smooth_ic = self.options.smooth_ic
        self.dxl = (self.x0 - self.xmin) / self.nl
        ratio = self.rhor / self.rhol
        self.dxr = self.dxl / ratio
        self.h0 = self.hdx * self.dxr

    def create_particles(self):
        return self.generate_particles(
            xmin=self.xmin, xmax=self.xmax, x0=self.x0, rhol=self.rhol,
            rhor=self.rhor, pl=self.pl, pr=self.pr, bx=0.03, gamma1=gamma1,
            ul=self.ul, ur=self.ur, dxl=self.dxl, dxr=self.dxr, h0=self.h0
        )


    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=1.0, eps=0.5, g1=0.2, g2=0.4)

        # Fix mpm scheme first
        mpm = GasDScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=20, tol=1e-6
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            hfact=1.2
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            hfact=1.2)

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            ndes=7, recycle_accelerations=False)

        s = SchemeChooser(default='adke', adke=adke, gsph=gsph,
                          psph=psph, tsph=tsph, magma2=magma2)
        return s


if __name__ == '__main__':
    app = Blastwave()
    app.run()
    app.post_process()
