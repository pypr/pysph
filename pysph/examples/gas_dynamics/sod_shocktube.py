"""Simulate the classical Sod Shocktube problem in 1D (5 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import (ADKEScheme, GasDScheme, GSPHScheme,
                              SchemeChooser, add_bool_argument)
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.base.nnps import DomainManager

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.15


class SodShockTube(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = 0.0
        self.rhol = 1.0
        self.rhor = 0.125
        self.pl = 1.0
        self.pr = 0.1
        self.ul = 0.0
        self.ur = 0.0

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=640,
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
        self.smooth_ic = self.options.smooth_ic
        self.dscheme = self.options.dscheme
        self.dxl = (self.x0 - self.xmin) / self.nl
        if self.dscheme == 'constant_mass':
            ratio = self.rhor / self.rhol
            self.dxr = self.dxl / ratio
        else:
            self.dxr = self.dxl
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx
        self.dt = dt
        self.tf = tf

    def create_particles(self):
        # Boundary particles are not needed as we are mirroring the particles
        # using the domain manager. Hence, bx is set to 0.0.
        f, b = self.generate_particles(
            xmin=self.xmin, xmax=self.xmax, x0=self.x0, rhol=self.rhol,
            rhor=self.rhor, pl=self.pl, pr=self.pr, bx=0.00, gamma1=gamma1,
            ul=self.ul, ur=self.ur, dxl=self.dxl, dxr=self.dxr, h0=self.h0
        )
        self.scheme.setup_properties([f, b])
        return [f]

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, mirror_in_x=True,
            n_layers=2
        )

    def configure_scheme(self):
        scheme = self.scheme
        if self.options.scheme in ['gsph', 'mpm']:
            scheme.configure(kernel_factor=self.hdx)
        elif self.options.scheme in ['psph', 'tsph']:
            scheme.configure(hfact=self.hdx)
        scheme.configure_solver(tf=self.tf, dt=self.dt)

    def create_scheme(self):
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1.0, k=0.3, eps=0.5, g1=0.2, g2=0.4)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=None, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
        )
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=None,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=True, blend_alpha=2.0,
            niter=20, tol=1e-6
        )
        crk = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0,
            nu=0, h0=0, p0=0, gamma=gamma, cl=3
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=None
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=None
        )

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            has_ghosts=True, ndes=7
        )

        s = SchemeChooser(
            default='adke', adke=adke, mpm=mpm, gsph=gsph, crk=crk, psph=psph,
            tsph=tsph, magma2=magma2)
        return s


if __name__ == '__main__':
    app = SodShockTube()
    app.run()
    app.post_process()
