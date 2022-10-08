"""Simulate the Robert's problem (1D) (40 seconds).
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
        group.add_argument(
            "--dscheme", choices=["constant_mass", "constant_volume"],
            dest="dscheme", default="constant_mass",
            help="Spatial discretization scheme, one of {'constant_mass', "
                 "'constant_volume'}."
        )
        add_bool_argument(group, 'smooth-ic', dest='smooth_ic', default=False,
                          help="Smooth the initial condition.")

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        self.dscheme = self.options.dscheme
        self.xb_ratio = 2
        self.dxl = (self.x0 - self.xmin) / self.nl
        if self.dscheme == 'constant_mass':
            ratio = self.rhor / self.rhol
            self.dxr = self.dxl / ratio
        else:
            self.dxr = self.dxl
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        return self.generate_particles(
            xmin=self.xmin * self.xb_ratio, xmax=self.xmax * self.xb_ratio,
            x0=self.x0, rhol=self.rhol, dxl=self.dxl, dxr=self.dxr,
            rhor=self.rhor, pl=self.pl, pr=self.pr, bx=0.03, gamma1=gamma1,
            ul=self.ul, ur=self.ur, h0=self.h0
        )

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=2.0, k=1.0, eps=0.5, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=2.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6
        )

        # PSPH doesn't work with default number of particles due to particle
        # penetration. Reduce the number of particles, say with --nl=1500
        # for this example to work with PSPH scheme.
        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=1.2
        )

        # TSPH doesn't work with default number of particles due to particle
        # penetration. Reduce the number of particles, say with --nl=1500
        # for this example to work with TSPH scheme.
        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=1.2
        )

        # MAGMA2 doesn't work with default parameters for this problem. Need
        # to use --timestep=0.5e-4.
        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            adaptive_h_scheme='mpm', hfact=1.2, recycle_accelerations=False
        )
        s = SchemeChooser(default='adke', adke=adke, mpm=mpm, gsph=gsph,
                          psph=psph, tsph=tsph, magma2=magma2)
        return s


if __name__ == '__main__':
    app = Robert()
    app.run()
    app.post_process()
