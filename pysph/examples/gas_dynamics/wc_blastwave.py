"""Woodward Collela blastwave (2 minutes)

Two discontinuities moving towards each other and the
results after they interact
"""
from pysph.sph.scheme import (
    GSPHScheme, SchemeChooser, ADKEScheme, GasDScheme
)
from pysph.sph.wc.crksph import CRKSPHScheme
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
            name='fluid', x=x, rho=self.rho, p=p, h=h, m=m, e=e, cs=cs,
            h0=h, u=0
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
            alpha=1, beta=1.0, k=1.0, eps=0.8, g1=0.2, g2=0.4,
            has_ghosts=True)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
            has_ghosts=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=20, tol=1e-6, has_ghosts=True
        )

        crk = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=gamma, cl=4, cq=1, eta_crit=0.2, has_ghosts=True
        )

        s = SchemeChooser(
            default='gsph', gsph=gsph, adke=adke, mpm=mpm, crksph=crk
            )
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
        elif self.options.scheme == 'crksph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=20)

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

        plot_exact = False
        plot_legends = ['pysph(%s)' % (self.options.scheme)]

        try:
            import h5py
            plot_exact = True
            plot_legends.append('exact')
        except ImportError:
            print("h5py not found, exact data will not be plotted")

        fname = os.path.join(
            os.path.dirname(__file__), 'wc_exact.hdf5'
        )
        props = ["rho", "u", "p", "e"]
        props_h5 = ["'" + pr + "'" for pr in props]
        if plot_exact:
            h5file = h5py.File(fname)
            dataset = h5file['data_0']

        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']
        x = pa.x
        u = pa.u
        e = pa.e
        p = pa.p
        rho = pa.rho
        prop_vals = [rho, u, p, e]
        for _i, prop in enumerate(props):
            pyplot.plot(x, prop_vals[_i])
            if plot_exact:
                pyplot.scatter(
                    dataset.get(props_h5[_i])['data_0'].get("'x'")
                    ['data_0'][:],
                    dataset.get(props_h5[_i])['data_0'].get("'data'")
                    ['data_0'][:],
                    c='k', s=4
                    )
            pyplot.xlabel('x')
            pyplot.ylabel(props[_i])
            pyplot.legend(plot_legends)
            fig = os.path.join(self.output_dir, props[_i] + ".png")
            pyplot.savefig(fig, dpi=300)
            pyplot.close('all')


if __name__ == '__main__':
    app = WCBlastwave()
    app.run()
    app.post_process()
