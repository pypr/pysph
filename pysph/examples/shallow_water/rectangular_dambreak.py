"""Rectangular dam break over a dry bed. (8 mins)

The case is described in "A corrected smooth particle hydrodynamics formulation
of the shallow-water equations", Miguel Rodriguez-Paz and Javier Bonet,
Computers & Structures, Vol 83, pp 1396-1410 (2005).
DOI:10.1016/j.compstruc.2004.11.025

"""
import os

# Numpy
from numpy import (array, ones_like, mgrid, sort, linspace, arange,
                   concatenate, savez)

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
        InitialGuessDensity, SummationDensity, DensityResidual,
        DensityNewtonRaphsonIteration, CheckConvergence,
        UpdateSmoothingLength, SWEOS, SWEIntegrator, SWEStep,
        CorrectionFactorVariableSmoothingLength, ParticleAcceleration
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class RectangularDamBreak(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.025,
            help="Spacing between the particles")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=1.0,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--le", action="store", type=float, dest="le", default=2.0,
            help="Initial length of the fluid column (m)")
        group.add_argument(
            "--w", action="store", type=float, dest="w", default=1.0,
            help="Initial width of the fluid column (m)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.dw0 = self.options.dw0
        self.le = self.options.le
        self.w = self.options.w

    def create_particles(self):
        """Create the Rectangular patch of fluid."""
        # The wall at x = 0 is simulated by applying a symmetry condition
        # in the fluid, i.e., an additional column of fluid to the left of the
        # wall
        x, y = mgrid[-self.le:self.le+1e-4:self.dx,
                     -(self.w)/2.:(self.w)/2.+1e-4:self.dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x) * self.dx * self.dx * rho_w * self.dw0
        h = ones_like(x) * self.hdx * self.dx
        h0 = ones_like(x) * self.hdx * self.dx

        rho = ones_like(x) * rho_w * self.dw0
        rho0 = ones_like(x) * rho_w * self.dw0

        pa = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                     name='fluid')

        compute_initial_props([pa])
        return [pa]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 1.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=[0.1, 0.4, 0.7, 1.0],
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim=dim, dest='fluid',
                                        sources=['fluid']),
                    UpdateSmoothingLength(dim=dim, dest='fluid')
                ], update_nnps=True
            ),

            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid']),
                    SummationDensity(dest='fluid', sources=['fluid']),
                    DensityResidual('fluid')
                ]
            ),

            Group(
                equations=[
                    Group(
                        equations=[
                            DensityNewtonRaphsonIteration(dim=dim,
                                                          dest='fluid'),
                            UpdateSmoothingLength(dim=dim, dest='fluid')
                        ], update_nnps=True
                    ),

                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(
                                dest='fluid', sources=['fluid']),
                            SummationDensity(dest='fluid', sources=['fluid']),
                            DensityResidual(dest='fluid'),
                            CheckConvergence(dest='fluid')
                        ],
                    )
                ], iterate=True, max_iterations=10
            ),

            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(dest='fluid', 
                                                            sources=['fluid']),
                    SWEOS(dest='fluid')
                ]
            ),

            Group(
                equations=[
                    ParticleAcceleration(dim=dim, dest='fluid', 
                                         sources=['fluid'], u_only=True)
                ]
            ),
        ]
        return equations

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        import matplotlib
        matplotlib.use('Agg')
        self._plot_depth_of_dam()

    def _plot_depth_of_dam(self):
        from matplotlib import pyplot as plt
        t_to_plot = [0.1, 0.4, 0.7, 1.0]
        fname_for_plot = []
        for fname in self.output_files:
            data = load(fname)
            t = data['solver_data']['t']
            if t in t_to_plot:
                fname_for_plot.append(fname)

        def pos_actual(dw0, dw, t):
            # Given the initial depth (dw0) and depth (dw) of a fluid column
            # section at time t, returns the exact location of that section
            # with respect to the gate position (Ritters Solution)
            # Note: This solution is valid till the expansion wave reaches the
            # wall
            g = 9.8
            return (2*(g*dw0)**0.5 - 3*(g*dw)**0.5)*t

        num = 0
        for fname in fname_for_plot:
            data = load(fname)
            fluid = data['arrays']['fluid']

            y = arange(-(self.w)/2., (self.w)/2.+1e-4, self.dx)
            y_srt = sort(y)
            len_y = len(y_srt)
            idx_mid_y = int(len_y/2.)
            mid_y = y_srt[idx_mid_y]

            # Mid section of fluid column along positive x axis
            cond1 = abs(fluid.y - mid_y) < 1e-10
            cond2 = fluid.x >= 0
            cond = cond1 & cond2

            x = fluid.x[cond]
            dw = fluid.dw[cond]
            u = fluid.u[cond]
            rho = fluid.rho[cond]
            au = fluid.au[cond]

            if len(x) == 0:
                return

            x_ind_sort = x.argsort()
            x = x[x_ind_sort]
            x_max = max(x)
            dw = dw[x_ind_sort]
            u = u[x_ind_sort]
            rho = rho[x_ind_sort]
            au = au[x_ind_sort]

            # To get the remaining analytical curve of pos vs depth
            dw_rem = linspace(dw[-1], 0, 50)
            dw_actual = concatenate((dw, dw_rem), axis=0)

            x_actual = pos_actual(self.dw0, dw_actual, t_to_plot[num])
            x_actual = concatenate((array([0]), x_actual+self.le), axis=0)

            dw_actual = concatenate(
                    (array([self.options.dw0]), dw_actual), axis=0)

            fname_res = os.path.join(
                    self.output_dir, 'results%d.npz' % (num+1))
            savez(
                fname_res, t=t_to_plot[num], x=x, x_actual=x_actual, y=y,
                dw=dw, dw_actual=dw_actual, u=u, rho=rho, au=au)

            plt.clf()
            plt.plot(x_actual, dw_actual, 'r-', label='Analytical Solution',
                     linewidth=2.0)
            plt.plot(x, dw, 'bo', label='SWE-SPH', markersize=2.5)
            plt.ylim(0, 1.2*self.dw0)
            plt.xlim(0, 1.5*x_max)
            plt.legend(prop={'size': 13})
            plt.xlabel('x (m)', fontsize=13)
            plt.ylabel('dw (m)', fontsize=13)
            plt.tick_params(axis='x', labelsize=13)
            plt.tick_params(axis='y', labelsize=13)
            plt.title('t = %0.1f s' % t_to_plot[num], fontsize=13)
            fig = os.path.join(self.output_dir, "depth%d.png" % (num+1))
            plt.savefig(fig, dpi=300)
            num += 1


def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                        sources=['fluid']),
                SWEOS(dest='fluid')
            ]
        )
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreak()
    app.run()
    app.post_process(app.info_filename)
