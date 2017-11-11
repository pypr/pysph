import numpy as np
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep


class EllipticalDrop(Application):
    def create_particles(self):
        dx = 0.025
        x, y = np.mgrid[-1.05:1.05:dx, -1.05:1.05:dx]
        mask = x*x + y*y < 1
        x = x[mask]
        y = y[mask]
        rho = 1.0
        h = 1.3*dx
        m = rho*dx*dx
        pa = get_particle_array_wcsph(
            name='fluid', x=x, y=y, u=-100*x, v=100*y, rho=rho,
            m=m, h=h
        )
        return [pa]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    TaitEOS(dest='fluid', sources=None, rho0=1.0,
                            c0=1400, gamma=7.0),
                ],
                real=False
            ),

            Group(
                equations=[
                    ContinuityEquation(dest='fluid',  sources=['fluid']),

                    MomentumEquation(dest='fluid', sources=['fluid'],
                                     alpha=0.1, beta=0.0, c0=1400),

                    XSPHCorrection(dest='fluid', sources=['fluid']),
                ]
            ),
        ]
        return equations

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = PECIntegrator(fluid=WCSPHStep())

        dt = 5e-6
        tf = 0.0076
        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator,
            dt=dt, tf=tf
        )
        return solver


if __name__ == '__main__':
    app = EllipticalDrop(fname='ed_no_scheme')
    app.run()
