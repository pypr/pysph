""" Dam break using PCISPH
"""

from __future__ import print_function
import numpy as np

# geometry
from pysph.examples._db_geometry import create_2D_filled_region
from pysph.examples.dam_break_2d import DamBreak2DGeometry

# PySPH base and carray imports
from pysph.solver.application import Application
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver

# equations
from pysph.sph.equation import Group
from pysph.sph.pcisph import (
    get_particle_array_pcisph, SaveCurrentProps, ClearAccelerationsAddGravity,
    SetUpPressureSolver, Advect, ComputeDensityFluid, DensityDifference,
    CorrectPressure, MomentumEquation, LeapFrogIntegrator, PCISPHStep)


class DamBreak2d(Application):
    def initialize(self):
        self.dx = 0.025
        self.hdx = 1.2
        self.h = self.hdx * self.dx
        self.rho = 1000
        self.u = 2
        self.m = 1000 * self.dx * self.dx
        self.dt = 5e-3

    def create_particles(self):
        """Create the circular patch of fluid."""
        xf1, yf1 = create_2D_filled_region(-1, 0, 0, 1, self.dx)
        xf2, yf2 = create_2D_filled_region(0.5, 0, 1.5, 1, self.dx)
        u1 = np.ones_like(xf1) * self.u
        u2 = np.ones_like(xf1) * -self.u

        xf, yf = np.concatenate([xf1, xf2]), np.concatenate([yf1, yf2])
        uf = np.concatenate([u1, u2])

        consts = {'rho_base': [1000.]}

        fluid = get_particle_array_pcisph(x=xf, y=yf, h=self.h, m=self.m,
                                          rho=self.rho, u=uf, name="fluid",
                                          constants=consts, dim=2, dt=self.dt)

        return [fluid]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = LeapFrogIntegrator(fluid=PCISPHStep())

        dt = self.dt
        print("DT: %s" % dt)
        tf = 1
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                SaveCurrentProps(dest='fluid', sources=None),
                ClearAccelerationsAddGravity(dest='fluid', sources=None),
                # Calculate densities to compute non-pressure forces
                # -----------------------------------------------
                # compute non pressure forces if any
                # -----------------------------------------------
                SetUpPressureSolver(dest='fluid', sources=None)
            ]),
            # pressure solve
            Group(
                equations=[
                    Group(equations=[
                        # In initialize of `Advect` equation, predict
                        # the particles position
                        Advect(dest='fluid', sources=None),
                        # In loop method of ComputeDensityFluid, compute
                        # the density of fluid.
                        ComputeDensityFluid(dest='fluid', sources=['fluid']),

                        # in post_loop of `DensityDifference` find the density
                        # density difference from base pressure
                        DensityDifference(dest='fluid', sources=None,
                                          debug=True),
                        # in post_loop of `CorrectPressure` correct
                        # the pressure by density difference
                        CorrectPressure(dest='fluid', sources=None)
                    ]),
                    Group(equations=[
                        # compute pressure acceleration
                        MomentumEquation(dest='fluid', sources=['fluid'])
                    ])
                ],
                iterate=True,
                min_iterations=1,
                max_iterations=4)
        ]
        return equations


if __name__ == '__main__':
    app = FluidsColliding()
    app.run()
