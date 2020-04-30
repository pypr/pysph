"""Wall-shock problem in 1D (40 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, GSPHScheme, SchemeChooser

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-6
tf = 0.4

# domain size and discretization parameters
xmin = -0.2
xmax = 0.2


class WallShock(ShockTubeSetup):

    def initialize(self):
        self.xmin = xmin
        self.xmax = xmax
        self.x0 = 0.0
        self.rhol = 1.0
        self.rhor = 1.0
        self.pl = 4e-7
        self.pr = 4e-7
        self.ul = 1.0
        self.ur = -1.0

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.5,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=500,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor/self.rhol
        self.nr = ratio*self.nl
        self.xb_ratio = 5
        self.dxl = (self.x0 - self.xmin) / self.nl
        self.dxr = (self.xmax - self.x0) / self.nr
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        return self.generate_particles(xmin=self.xmin*self.xb_ratio,
                                       xmax=self.xmax*self.xb_ratio,
                                       dxl=self.dxl, dxr=self.dxr,
                                       m=self.dxl, pl=self.pl,
                                       pr=self.pr, h0=self.h0, bx=0.02,
                                       gamma1=gamma1, ul=self.ul, ur=self.ur)

    def create_scheme(self):
        self.dt = dt
        self.tf = tf

        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=0.7, eps=0.5, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6
        )

        s = SchemeChooser(default='adke', adke=adke, mpm=mpm, gsph=gsph)
        return s


if __name__ == '__main__':
    app = WallShock()
    app.run()
    app.post_process()
