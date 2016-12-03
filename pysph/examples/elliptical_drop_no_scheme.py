"""Evolution of a circular patch of incompressible fluid. (60 seconds)

This shows how one can explicitly setup equations and the solver instead of
using a scheme.
"""
from __future__ import print_function
from numpy import ones_like, mgrid

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import Gaussian

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation

from pysph.examples.elliptical_drop import EllipticalDrop as EDScheme


class EllipticalDrop(EDScheme):
    def create_scheme(self):
        # Don't create a scheme as done in the parent example class.
        return None

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        ro = self.ro
        name = 'fluid'
        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        # Get the particles inside the circle.
        condition = ~((x*x + y*y - 1.0) > 1e-10)
        x = x[condition].ravel()
        y = y[condition].ravel()

        m = ones_like(x)*dx*dx*ro
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) * ro
        u = -100*x
        v = 100*y

        pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h,
                                      u=u, v=v, name=name)

        print("Elliptical drop :: %d particles" %
              (pa.get_number_of_particles()))

        # add requisite variables needed for this formulation
        for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
                     'v0', 'w0', 'x0', 'y0', 'z0'):
            pa.add_property(name)

        # set the output property arrays
        pa.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm',
                              'h', 'p', 'pid', 'tag', 'gid'])

        return [pa]

    def create_solver(self):
        print("Create our own solver.")
        kernel = Gaussian(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep())

        dt = 5e-6
        tf = 0.0076
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=True,
                        cfl=0.3, n_damp=50,
                        output_at_times=[0.0008, 0.0038])

        return solver

    def create_equations(self):
        print("Create our own equations.")
        equations = [
            Group(
                equations=[
                    TaitEOS(
                        dest='fluid', sources=None, rho0=self.ro,
                        c0=self.co, gamma=7.0
                    ),
                ],
                real=False
            ),
            Group(equations=[
                ContinuityEquation(dest='fluid',  sources=['fluid']),

                MomentumEquation(
                    dest='fluid', sources=['fluid'],
                    alpha=self.alpha, beta=0.0, c0=self.co
                ),

                XSPHCorrection(dest='fluid', sources=['fluid']),

            ]),
        ]
        return equations


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
    app.post_process(app.info_filename)
