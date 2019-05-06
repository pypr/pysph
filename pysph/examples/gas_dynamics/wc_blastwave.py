"""Simulate the classical Sod Shocktube problem in 1D (5 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, GSPHScheme, SchemeChooser
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 5e-6
tf = 0.038


class WCBlastwave(Application):
    def initialize(self):
        self.xmin = 0.0
        self.xmax = 1.0
        self.domain_length = self.xmax - self.xmin
        self.rho = 1.0
        self.p1 = 1000
        self.p2 = 0.01
        self.p3 = 100
        self.u = 0.0
        self.gamma = gamma
        self.hdx = 1.5
        self.n_particles = 1000

    def consume_user_options(self):
        pass

    def create_particles(self):
        self.dx = self.domain_length / self.n_particles
        x = numpy.arange(
            self.xmin + self.dx*0.5, self.xmax, self.dx
            )

        p = numpy.ones_like(x) * self.p2

        left_indices = numpy.where(x < 0.1)[0]
        right_indices = numpy.where(x > 0.9)[0]

        p[left_indices] = self.p1
        p[right_indices] = self.p3

        h = self.hdx*self.dx
        m = self.dx * self.rho
        e = p / ((self.gamma - 1) * self.rho)

        cs = numpy.sqrt(self.gamma * p / self.rho)

        fluid = gpa(
            name='fluid', x=x, rho=self.rho, p=p, h=h, m=m, e=e, cs=cs
            )

        self.scheme.setup_properties([fluid])
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax,
            mirror_in_x=True
        )

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1.0, k=1.0, eps=0.8, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=20, tol=1e-6
        )
        s = SchemeChooser(default='gsph', adke=adke, mpm=mpm, gsph=gsph)
        return s
    
    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
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
        u = pa.u
        e = pa.e
        p = pa.p
        rho = pa.rho
        cs = pa.cs
        
        pyplot.plot(x, rho)
        pyplot.xlabel('x')
        pyplot.ylabel('rho')
        fig = os.path.join(self.output_dir, "density.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.close('all')

        pyplot.plot(x, u)
        pyplot.xlabel('x')
        pyplot.ylabel('u')
        fig = os.path.join(self.output_dir, "velocity.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.close('all')

        pyplot.plot(x, p)
        pyplot.xlabel('x')
        pyplot.ylabel('p')
        fig = os.path.join(self.output_dir, "pressure.png")
        pyplot.savefig(fig, dpi=300)
        pyplot.close('all')


if __name__ == '__main__':
    app = WCBlastwave()
    app.run()
    app.post_process()
