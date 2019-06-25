"""Simulate the hydrostatic box problem (30 minutes)

high density square region inside a low density square medium, in pressure
equilibrium, the solution should not evolve in time
"""

from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.sph.scheme import (
    ADKEScheme, SchemeChooser, GasDScheme, GSPHScheme
)
from pysph.base.utils import get_particle_array as gpa
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.tools import uniform_distribution as ud
import numpy


class HydrostaticBox(Application):
    def initialize(self):
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0
        self.gamma = 1.5
        self.p = 1
        self.rho0 = 1
        self.rhoi = 4
        self.nx = 50
        self.ny = self.nx
        self.dx = (self.xmax - self.xmin) / self.nx
        self.hdx = 1.5
        self.dt = 1e-3
        self.tf = 10

    def create_particles(self):
        data = ud.uniform_distribution_cubic2D(
            self.dx, self.xmin, self.xmax, self.ymin, self.ymax
        )
        x = data[0]
        y = data[1]
        box_indices = numpy.where(
            (x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75)
        )[0]
        rho = numpy.ones_like(x) * self.rho0
        rho[box_indices] = self.rhoi
        e = self.p / ((self.gamma - 1) * rho)
        m = self.dx * self.dx * rho
        h = self.hdx * self.dx

        fluid = gpa(
            name='fluid', x=x, y=y, p=self.p, rho=rho, e=e, u=0., v=0.,
            h=self.hdx*self.dx, m=m, h0=h
        )

        self.scheme.setup_properties([fluid])
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax,
            periodic_in_x=True, periodic_in_y=True
        )

    def create_scheme(self):
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=2,
            gamma=self.gamma, kernel_factor=1.0,
            g1=0., g2=0., rsolver=7, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=5.0,
            niter=40, tol=1e-6, has_ghosts=True
        )

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=2, gamma=self.gamma,
            kernel_factor=1.2, alpha1=0, alpha2=0,
            beta=2.0, update_alpha1=False, update_alpha2=False,
            has_ghosts=True
        )

        crk = CRKSPHScheme(
            fluids=['fluid'], dim=2, rho0=0, c0=0, nu=0, h0=0, p0=0,
            gamma=self.gamma, cl=2, has_ghosts=True
        )
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=2, gamma=self.gamma,
            alpha=0.1, beta=0.1, k=1.5, eps=0., g1=0.1, g2=0.1,
            has_ghosts=True)
        s = SchemeChooser(
            default='crksph', crksph=crk, adke=adke, mpm=mpm, gsph=gsph
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        if self.options.scheme == 'gsph':
            s.configure_solver(
                dt=self.dt, tf=self.tf,
                adaptive_timestep=True, pfreq=50
            )
        elif self.options.scheme == 'mpm':
            s.configure(kernel_factor=1.2)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'crksph':
            s.configure_solver(
                dt=self.dt, tf=self.tf, adaptive_timestep=False, pfreq=50
            )
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)


if __name__ == "__main__":
    app = HydrostaticBox()
    app.run()
