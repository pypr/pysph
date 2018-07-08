"""Particle splitting in a localized region in a square domain to determine the
error in density field after splitting. (0.06 s)

The case is described in "Accurate particle splitting for smoothed particle
hydrodynamics in shallow water with shock capturing", R. Vacondio, B. D. Rogers
and P.K. Stansby, Int. J. Numer. Meth. Fluids, Vol 69, pp 1377-1410 (2012).
DOI: 10.1002/fld.2646

"""
# Numpy
from numpy import (ones_like, mgrid, array, sqrt)
import numpy as np

# PySPH base
from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_swe as gpa_swe

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH equations
from pysph.sph.equation import Group
from pysph.sph.swe.basic import (
        GatherDensityEvalNextIteration, NonDimensionalDensityResidual, SWEStep,
        UpdateSmoothingLength, CheckConvergenceDensityResidual, SWEIntegrator,
        InitialGuessDensityVacondio, ParticleSplit, CheckForParticlesToSplit
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 10000.0
g = 9.81
dim = 2


class ParticleSplitTest(Application):
    def create_particles(self):
        # Fluid particles
        hdx = 1.0
        d = 1.0
        dx = 50
        len_fluid_domain = 1400

        x, y = mgrid[0:len_fluid_domain+1e-4:dx, 0:len_fluid_domain+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        idx_inner_pa_to_split = []
        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            if (
                (6*dx <= x_i <= len_fluid_domain-6*dx) and
                (6*dx <= y_i <= len_fluid_domain-6*dx)
            ):
                idx_inner_pa_to_split.append(idx)
        idx_inner_pa_to_split = array(idx_inner_pa_to_split)

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        A = m / rho
        A[idx_inner_pa_to_split] = 3000

        pa = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h=h, h0=h0, A=A,
                     name='fluid')

        # Boundary Particles
        x, y = mgrid[-2*dx:len_fluid_domain+2*dx+1e-4:dx,
                     -2*dx:len_fluid_domain+2*dx+1e-4:dx]
        x = x.ravel()
        y = y.ravel()
        boun_idx = np.where((x < 0) | (y < 0) | (x > len_fluid_domain) | (y >
                            len_fluid_domain))
        x = x[boun_idx]
        y = y[boun_idx]
        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        rho = ones_like(x) * rho_w * d

        boundary = gpa_swe(name='boundary', x=x, y=y, m=m, h=h, rho=rho)

        compute_initial_props([pa])
        return [pa, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        dt = 1e-4
        tf = 1e-4
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            dt=dt,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            InitialGuessDensityVacondio(dim=dim, dest='fluid',
                                                        sources=['fluid']),
                            ]
                        ),
                    Group(
                        equations=[
                           GatherDensityEvalNextIteration(
                               dest='fluid', sources=['fluid', 'boundary']),
                            ]
                        ),
                    Group(
                        equations=[
                            NonDimensionalDensityResidual(dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(dim=dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            CheckConvergenceDensityResidual(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=10
            ),
        ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.nnps.update()

    def post_process(self):
        rho_exact = 1e4
        rho_num = self.particles[0].rho
        print('\nMax rho is %0.3f' % max(rho_num))
        l2_err_rho = sqrt(np.sum((rho_exact - rho_num)**2)
                          / len(rho_num))
        print('L2 error in density is %0.3f \n' % l2_err_rho)


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        CheckForParticlesToSplit(
                            dest='fluid', A_max=2900, x_min=300, x_max=1100,
                            y_min=300, y_max=1100)
                            ],
                    )
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = ParticleSplitTest()
    app.run()
    app.post_process()
