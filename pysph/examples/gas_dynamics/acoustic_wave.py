"""
Propagation of acoustic wave
particles are have properties according
to the following distribuion
\rho = \rho_0 + \Delta\rho sin(kx)
p = p_0 + c_0^2\Delta\rho sin(kx)
u = c_0\rho_0^{-1}\Delta\rho sin(kx)

with \Delta\rho = 1e-6 and k = 2\pi/\lambda
where \lambda is the domain length.
\rho_0 = \gamma = 1.4 and p_0 = 1.0
"""


# standard library and numpy imports
from __future__ import print_function
import numpy

#pysph imports
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import \
    GSPHScheme, ADKEScheme, GasDScheme, SchemeChooser

class AcousticWave(Application):
    def initialize(self):
        self.xmin = 0.
        self.xmax = 1.
        self.gamma = 1.4
        self.rho_0 = self.gamma
        self.p_0 = 1.
        self.c_0 = 1.
        self.delta_rho = 1e-6
        self.n_particles = 400
        self.l = self.xmax - self.xmin
        self.dx = self.l / (self.n_particles)
        self.k = -2 * numpy.pi / self.l
        self.hdx = 1.
        self.dt = 1e-3
        self.tf = 10
        self.dim = 1

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=1, periodic_in_x=True
        )

    def create_particles(self):
        x = numpy.arange(
            self.xmin + self.dx*0.5, self.xmax, self.dx
        )
        rho = self.rho_0 + self.delta_rho *\
            numpy.sin(self.k * x)

        p = self.p_0 + self.c_0**2 *\
            self.delta_rho * numpy.sin(self.k * x)

        u = self.c_0 * self.delta_rho * numpy.sin(self.k * x) /\
            self.rho_0
        cs = numpy.sqrt(
            self.gamma * p / rho
        )
        h = numpy.ones_like(x) * self.dx * self.hdx
        m = numpy.ones_like(x) * self.dx * rho
        e = p / ((self.gamma - 1) * rho)
        fluid = gpa(
            name='fluid', x=x, p=p, rho=rho, u=u, h=h, m=m, e=e, cs=cs
        )
        self.scheme.setup_properties([fluid])

        return [fluid,]

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=self.dim, 
            gamma=self.gamma, kernel_factor=1.0,
            g1=0., g2=0., rsolver=7, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=5.0,
            niter=40, tol=1e-6
        )

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=self.dim, gamma=self.gamma,
            kernel_factor=1.2, alpha1=0, alpha2=0,
            beta=2.0, update_alpha1=False, update_alpha2=False
        )

        s = SchemeChooser(
            default='gsph', gsph=gsph, mpm=mpm
        )

        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(
                dt=self.dt, tf=self.tf,
                adaptive_timestep=True, pfreq=50
            )

        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        
    def post_process(self):
        from pysph.solver.utils import load
        if len(self.output_files) < 1:
            return
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']
        x_c = pa.x
        u = self.c_0 * self.delta_rho * numpy.sin(self.k * x_c) /\
            self.rho_0
        u_c = pa.u
        l_inf = numpy.max(
            numpy.abs(u_c - u)
        )

        print("L_inf norm for the problem: %s" %(l_inf))

if __name__ == "__main__":
    app = AcousticWave()
    app.run()
    app.post_process()