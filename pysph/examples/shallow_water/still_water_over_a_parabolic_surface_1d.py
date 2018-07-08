"""Still water over a parabolic surface. (0.4 sec)

The case is described in "Smoothed Particle Hydrodynamics: Approximate
zero-consistent 2-D boundary conditions and still shallow-water tests",
R. Vacondio, B. D. Rogers and P. K. Stansby, Int. J. Numer. Meth. Fluids,
Vol 69, pp 226-253 (2012). DOI: 10.1002/fld.2559

"""
import os

# Numpy
from numpy import (ones_like, zeros_like, savez, arange)

# PySPH base
from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_swe as gpa_swe

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.solver.utils import load

# PySPH equations
from pysph.sph.equation import Group
from pysph.sph.swe.basic import (
        GatherDensityEvalNextIteration, NonDimensionalDensityResidual,
        UpdateSmoothingLength, CheckConvergenceDensityResidual, SWEOS,
        SWEIntegrator, SWEStep, CorrectionFactorVariableSmoothingLength,
        ParticleAcceleration, FluidBottomElevation, FluidBottomGradient,
        FluidBottomCurvature, GradientCorrectionPreStep, GradientCorrection,
        BedGradient, BedCurvature
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 1


class StillWaterInaParabolicSuface(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=20.0,
            help="Spacing between the particles")
        group.add_argument(
            "--fluid_surf_height", action="store", type=float,
            dest="fluid_surf_hei", default=10.0, help="Surface height of fluid\
            column (m)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.fluid_surf_hei = self.options.fluid_surf_hei

    def create_particles(self):
        # 1D
        hdx = self.hdx
        dx = self.dx
        fluid_surf_hei = self.fluid_surf_hei

        # Bed
        l_bed = 8000.0
        dxb = dx
        xb = arange(0, l_bed+1e-4, dxb)
        bo = 10.0
        a = 3000.0
        b = bo * ((xb-0.5*l_bed)/a)**2

        Vb = ones_like(xb) * dxb
        hb = ones_like(xb) * hdx * dxb

        bed = gpa_swe(name='bed', x=xb, V=Vb, b=b, h=hb)

        # For gradient correction
        len_b = len(bed.x) * 9
        bed.add_constant('m_mat', [0.0] * len_b)

        # Fluid
        x = arange(1000+2*dx, 7000-2*dx+1e-4, dx)
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        fluid = gpa_swe(x=x, h=h, h0=h0, name='fluid')
        compute_fluid_elevation([fluid, bed])

        dw = fluid_surf_hei - fluid.b
        rho = dw * rho_w
        rho0 = dw * rho_w
        m = rho * dx

        fluid.m = m
        fluid.rho = rho
        fluid.rho0 = rho0
        fluid.dw = dw

        compute_initial_props([fluid])

        return [fluid, bed]

    def create_solver(self):
        kernel = CubicSpline(dim=1)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 10
        solver = Solver(
            kernel=kernel,
            dim=1,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            tf=tf
            )
        return solver

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        import matplotlib
        matplotlib.use('Agg')
        self._plot_depth_and_vel()

    def _plot_depth_and_vel(self):
        from matplotlib import pyplot as plt
        t_to_plot = [10.0]
        fname_for_plot = []
        for fname in self.output_files:
            data = load(fname)
            t = data['solver_data']['t']
            if t in t_to_plot:
                fname_for_plot.append(fname)

        num = 0
        for fname in fname_for_plot:
            data = load(fname)
            fluid = data['arrays']['fluid']
            t = data['solver_data']['t']

            x = fluid.x
            x_max = max(x)
            dw = fluid.dw
            dw_max = max(dw)
            u = fluid.u
            u_max = max(u)
            b = fluid.b
            numerical_fluid_surf_hei = dw + b
            exact_fluid_surf_hei = ones_like(x) * self.fluid_surf_hei

            plt.clf()

            plt.plot(x, numerical_fluid_surf_hei, 'b', label='SWE-SPH')
            plt.plot(x, exact_fluid_surf_hei, 'r', label='Exact Solution')
            plt.plot(x, b, 'k--', label='Bottom Elevation')
            plt.ylim(-0.1*self.fluid_surf_hei, 1.5*dw_max)
            plt.xlim(0, x_max+20*self.dx)
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('dw (m)')
            plt.title('t = %0.1fs' % t_to_plot[num])
            fig = os.path.join(self.output_dir, "depth%d.png" % (num+1))
            plt.savefig(fig, dpi=300)

            plt.clf()

            exact_vel = zeros_like(x)

            fname_res = os.path.join(
                    self.output_dir, 'results.npz')
            savez(
                fname_res, t=t_to_plot[num], x=x, b=b, exact_vel=exact_vel,
                u=u, numerical_fluid_surf_hei=numerical_fluid_surf_hei,
                exact_fluid_surf_hei=exact_fluid_surf_hei)

            plt.plot(x, u, 'b', label='SWE-SPH')
            plt.plot(x, exact_vel, 'r', label='Exact Solution')
            plt.ylim(-1.5*u_max, 1.5*u_max)
            plt.xlim(0, x_max+20*self.dx)
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('u (m/s)')
            plt.title('t = %0.1fs' % t_to_plot[num])
            fig = os.path.join(self.output_dir, "velocity%d.png" % (num+1))
            plt.savefig(fig, dpi=300)
            num += 1

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(
                                dest='fluid', sources=['fluid']
                                ),
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
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid'])
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid')
                    ]
                ),
            Group(
                equations=[
                    FluidBottomElevation(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    FluidBottomGradient(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    FluidBottomCurvature(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(
                        dim=dim, dest='fluid', sources=['fluid'], u_only=True)
                    ],
                ),
            ]
        return equations


def compute_fluid_elevation(particles):
    one_time_equations = [
       Group(
            equations=[
                FluidBottomElevation(dest='fluid', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                GradientCorrectionPreStep(dest='bed', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                GradientCorrection(dest='bed', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                BedGradient(dest='bed', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                BedCurvature(dest='bed', sources=['bed'])
                ]
            ),
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                SWEOS(dest='fluid')
                    ]
            ),
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = StillWaterInaParabolicSuface()
    app.run()
    app.post_process(app.info_filename)
