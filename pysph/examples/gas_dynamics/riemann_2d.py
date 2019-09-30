"""2-d Riemann problem (3 hours)

four shock waves' interaction at the centerof the domain
"""

# numpy and standard imports
import numpy

# pysph imports
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import GSPHScheme, SchemeChooser, ADKEScheme, GasDScheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.examples.gas_dynamics.riemann_2d_config import R2DConfig

# current case from the al possible unique cases
case = 3

# config for current case
config = R2DConfig(case)
gamma = 1.4
gamma1 = gamma - 1
kernel_factor = 1.5
dt = 1e-4
dim = 2


class Riemann2D(Application):
    def initialize(self):
        # square domain
        self.dt = dt
        self.tf = config.endtime

    def add_user_options(self, group):
        group.add_argument(
            "--dscheme", choices=["constant_mass", "constant_volume"],
            dest="dscheme", default="constant_volume",
            help="spatial discretization scheme, one of constant_mass"
            "or constant_volume"
        )
        group.add_argument(
            "--nparticles", action="store", type=int,
            dest="nparticles", default=200
        )

    def consume_user_options(self):
        self.nx = self.options.nparticles
        self.ny = self.nx
        self.dx = (config.xmax - config.xmin) / self.nx

        # discretization function
        if self.options.dscheme == "constant_volume":
            self.dfunction = self.create_particles_constant_volume
        elif self.options.dscheme == "constant_mass":
            self.dfunction = self.create_particles_constant_mass

    def create_particles_constant_volume(self):
        dx = self.dx
        dx2 = dx * 0.5
        vol = dx * dx

        xmin = config.xmin
        ymin = config.ymin
        xmax = config.xmax
        ymax = config.ymax
        xmid = config.xmid
        ymid = config.ymid

        rho1, u1, v1, p1 = config.rho1, config.u1, config.v1, config.p1
        rho2, u2, v2, p2 = config.rho2, config.u2, config.v2, config.p2
        rho3, u3, v3, p3 = config.rho3, config.u3, config.v3, config.p3
        rho4, u4, v4, p4 = config.rho4, config.u4, config.v4, config.p4
        x, y = numpy.mgrid[xmin+dx2:xmax:dx, ymin+dx2:ymax:dx]

        x = x.ravel()
        y = y.ravel()

        u = numpy.zeros_like(x)
        v = numpy.zeros_like(x)

        # density and mass
        rho = numpy.ones_like(x)
        p = numpy.ones_like(x)

        for i in range(x.size):
            if x[i] <= xmid:
                if y[i] <= ymid:  # w3
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                    v[i] = v3
                else:            # w2
                    rho[i] = rho2
                    p[i] = p2
                    u[i] = u2
                    v[i] = v2
            else:
                if y[i] <= ymid:  # w4
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                    v[i] = v4
                else:            # w1
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                    v[i] = v1

        # thermal energy
        e = p/(gamma1 * rho)

        # mass
        m = vol * rho

        # smoothing length
        h = kernel_factor * (m/rho)**(1./dim)

        # create the particle array
        pa = gpa(name='fluid', x=x, y=y, m=m, rho=rho, h=h,
                 u=u, v=v, p=p, e=e, h0=h.copy())

        return pa

    def create_particles_constant_mass(self):
        dx = self.dx

        xmin = config.xmin
        ymin = config.ymin
        xmax = config.xmax
        ymax = config.ymax
        xmid = config.xmid
        ymid = config.ymid

        rho_max = config.rho_max
        nb4 = self.nx/4
        dx0 = (xmax - xmid)/nb4
        vol0 = dx0 * dx0
        m0 = rho_max * vol0

        # first quadrant
        vol1 = config.rho_max/config.rho1 * vol0
        dx = numpy.sqrt(vol1)
        dxb2 = 0.5 * dx
        x1, y1 = numpy.mgrid[xmid+dxb2:xmax:dx, ymid+dxb2:ymax:dx]
        x1 = x1.ravel()
        y1 = y1.ravel()

        u1 = numpy.ones_like(x1) * config.u1
        v1 = numpy.zeros_like(x1) * config.v1
        rho1 = numpy.ones_like(x1) * config.rho1
        p1 = numpy.ones_like(x1) * config.p1
        m1 = numpy.ones_like(x1) * m0
        h1 = numpy.ones_like(x1) * kernel_factor * (m1/rho1)**(0.5)

        # second quadrant
        vol2 = config.rho_max/config.rho2 * vol0
        dx = numpy.sqrt(vol2)
        dxb2 = 0.5 * dx
        x2, y2 = numpy.mgrid[xmid-dxb2:xmin:-dx, ymid+dxb2:ymax:dx]
        x2 = x2.ravel()
        y2 = y2.ravel()
        u2 = numpy.ones_like(x2) * config.u2
        v2 = numpy.ones_like(x2) * config.v2
        rho2 = numpy.ones_like(x2) * config.rho2
        p2 = numpy.ones_like(x2) * config.p2
        m2 = numpy.ones_like(x2) * m0
        h2 = numpy.ones_like(x2) * kernel_factor * (m2/rho2)**(0.5)

        # third quadrant
        vol3 = config.rho_max/config.rho3 * vol0
        dx = numpy.sqrt(vol3)
        dxb2 = 0.5 * dx
        x3, y3 = numpy.mgrid[xmid-dxb2:xmin:-dx, ymid-dxb2:ymin:-dx]
        x3 = x3.ravel()
        y3 = y3.ravel()

        u3 = numpy.ones_like(x3) * config.u3
        v3 = numpy.ones_like(x3) * config.v3
        rho3 = numpy.ones_like(x3) * config.rho3
        p3 = numpy.ones_like(x3) * config.p3
        m3 = numpy.ones_like(x3) * m0
        h3 = numpy.ones_like(x3) * kernel_factor * (m3/rho3)**(0.5)

        # fourth quadrant
        vol4 = config.rho_max/config.rho4 * vol0
        dx = numpy.sqrt(vol4)
        dxb2 = 0.5 * dx
        x4, y4 = numpy.mgrid[xmid+dxb2:xmax:dx, ymid-dxb2:ymin:-dx]
        x4 = x4.ravel()
        y4 = y4.ravel()

        u4 = numpy.ones_like(x4) * config.u4
        v4 = numpy.ones_like(x4) * config.v4
        rho4 = numpy.ones_like(x4) * config.rho4
        p4 = numpy.ones_like(x4) * config.p4
        m4 = numpy.ones_like(x4) * m0
        h4 = numpy.ones_like(x4) * kernel_factor * (m4/rho4)**(0.5)

        # concatenate the arrays
        x = numpy.concatenate([x1, x2, x3, x4])
        y = numpy.concatenate([y1, y2, y3, y4])
        p = numpy.concatenate([p1, p2, p3, p4])
        u = numpy.concatenate([u1, u2, u3, u4])
        v = numpy.concatenate([v1, v2, v3, v4])
        h = numpy.concatenate([h1, h2, h3, h4])
        m = numpy.concatenate([m1, m2, m3, m4])
        rho = numpy.concatenate([rho1, rho2, rho3, rho4])

        # derived variables
        e = p/((gamma-1.0) * rho)

        # create the particle array
        pa = gpa(
            name='fluid', x=x, y=y, m=m, rho=rho, h=h, u=u, v=v, p=p, e=e,
            h0=h.copy()
        )

        return pa

    def create_particles(self):
        fluid = self.dfunction()
        self.scheme.setup_properties([fluid])

        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=config.xmin, xmax=config.xmax,
            ymin=config.ymin, ymax=config.ymax,
            mirror_in_x=True, mirror_in_y=True
        )

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.5,
            g1=0.25, g2=0.5, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1.0, k=1.0, eps=0.5, g1=0.2, g2=0.4,
            has_ghosts=True)

        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=2, has_ghosts=True
        )

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
            has_ghosts=True
        )

        s = SchemeChooser(
            default='gsph', gsph=gsph, adke=adke, crksph=crksph, mpm=mpm
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'mpm':
            s.configure(kernel_factor=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)

    def post_process(self):
        if len(self.output_files) < 1 or self.rank > 0:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        from pysph.solver.utils import load
        import os
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']
        x = pa.x
        y = pa.y
        pyplot.scatter(x, y, s=1)
        pyplot.xlim((0.1, 0.6))
        pyplot.ylim((0.1, 0.6))
        fig = os.path.join(self.output_dir, "positions.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.close('all')


if __name__ == "__main__":
    app = Riemann2D()
    app.run()
    app.post_process()
