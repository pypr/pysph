""" Dam break using PCISPH
"""

from __future__ import print_function
import numpy as np

# geometry
from pysph.examples.dam_break_2d import DamBreak2DGeometry

# PySPH base and carray imports
from pysph.solver.application import Application
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator

# equations
from pysph.sph.equation import Group
from pysph.sph.pcisph import (
    get_particle_array_pcisph, get_particle_array_static_boundary,
    SaveCurrentProps, ClearAccelerationsAddGravity, SetUpPressureSolver,
    Advect, ComputeDensityFluid, ComputeDensitySolid, DensityDifference,
    CorrectPressure, MomentumEquation, MomentumEquationStaticBoundary,
    PCISPHStep)


class DamBreak(Application):
    def initialize(self):
        self.rho = 1000
        self.dim = 2
        self.dt = 1e-3

    def create_particles(self):
        model = DamBreak2DGeometry()
        [f, b] = model.create_particles()
        consts = {'rho_base': [1000.]}

        xf = f.x + 5 * model.dx
        yf = f.y + 5 * model.dy
        h = 1.0 * model.dx
        fluid = get_particle_array_pcisph(
            x=xf, y=yf, h=h, m=f.m, rho=self.rho, name="fluid",
            constants=consts, dim=self.dim, dt=self.dt, delta=None)

        V = model.dx**2
        boundary = get_particle_array_static_boundary(x=b.x, y=b.y, V=V,
                                                      name="boundary")

        return [fluid, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EulerIntegrator(fluid=PCISPHStep())

        dt = self.dt
        print("DT: %s" % dt)
        tf = 1
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                SaveCurrentProps(dest='fluid', sources=None),
                ClearAccelerationsAddGravity(dest='fluid', sources=None, gy=-9.81),
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
                        ComputeDensitySolid(dest='fluid',
                                            sources=['boundary']),

                        # in post_loop of `DensityDifference` find the density
                        # density difference from base pressure
                        DensityDifference(dest='fluid', sources=None,
                                          debug=False),
                        # in post_loop of `CorrectPressure` correct
                        # the pressure by density difference
                        CorrectPressure(dest='fluid', sources=None)
                    ]),
                    Group(equations=[
                        # compute pressure acceleration
                        MomentumEquation(dest='fluid', sources=['fluid']),
                        MomentumEquationStaticBoundary(dest='fluid',
                                                       sources=['boundary'])
                    ])
                ],
                iterate=True,
                min_iterations=1,
                max_iterations=8)
        ]
        return equations


if __name__ == '__main__':
    app = DamBreak()
    app.run()
