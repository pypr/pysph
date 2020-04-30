"""Rectangular dam break over a dry bed with particle splitting and merging
(19 min).

The case without particle splitting is described in "A corrected smooth
particle hydrodynamics formulation of the shallow-water equations", Miguel
Rodriguez-Paz and Javier Bonet, Computers & Structures, Vol 83, pp 1396-1410
(2005). DOI:10.1016/j.compstruc.2004.11.025

"""
import os

# Numpy
from numpy import (ones_like, mgrid, array, sort, linspace, arange,
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
        GatherDensityEvalNextIteration, NonDimensionalDensityResidual,
        UpdateSmoothingLength, CheckConvergenceDensityResidual, SWEOS,
        SWEIntegrator, SWEStep, CorrectionFactorVariableSmoothingLength,
        ParticleAcceleration, ParticleSplit, CheckForParticlesToSplit,
        DaughterVelocityEval, FindMergeable, InitialDensityEvalAfterMerge
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class RectangularDamBreakSplitandMerge(Application):
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
        group.add_argument(
            "--coeff_A_split", action="store", type=float,
            dest="coeff_A_max", default=1.0, help="Ratio of area of\
            particle to initial area after which the particle splitting is\
            activated")
        group.add_argument(
            "--coeff_h_split", action="store", type=float, dest="coeff_h_max",
            default=2.0, help="Ratio of smoothing length of particle to\
            initial smoothing length after which the particle splitting is\
            deactivated")
        group.add_argument(
            "--coeff_A_merge", action="store", type=float,
            dest="coeff_A_min", default=1.0, help="Ratio of area of\
            particle to initial area below which the particle merging is\
            activated")
        group.add_argument(
            "--x_min_split", action="store", type=float,
            dest="x_min_split", default=2.0, help="Position after which\
            splitting is activated")
        group.add_argument(
            "--x_max_split", action="store", type=float,
            dest="x_max_split", default=3.0, help="Position after which\
            splitting is deactivated")
        group.add_argument(
            "--x_min_merge", action="store", type=float,
            dest="x_min_merge", default=3.0, help="Position after which\
            merging is activated")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.dw0 = self.options.dw0
        self.le = self.options.le
        self.w = self.options.w

        self.coeff_A_max = self.options.coeff_A_max
        self.coeff_A_min = self.options.coeff_A_min
        self.coeff_h_max = self.options.coeff_h_max

        # Particle splitting activated when (A_i > A_max and h_i < h_max) and
        # (x_min_split < x_i < x_max_split)
        self.A_max = self.coeff_A_max * self.dx**2
        self.h_max = self.coeff_h_max * self.hdx * self.dx
        self.x_min_split = self.options.x_min_split
        self.x_max_split = self.options.x_max_split

        # Particle merging activated when A_i < A_min and x_i > x_min_merge
        self.A_min = self.coeff_A_min * self.dx**2
        self.x_min_merge = self.options.x_min_merge

    def create_particles(self):
        """Create the Rectangular patch of fluid."""
        # The wall at x = 0 is simulated by applying a symmetry condition
        # in the fluid, i.e., an additional column of fluid to the left of the
        # wall
        x, y = mgrid[-self.le:self.le+1e-4:self.dx,
                     -(self.w)/2.:(self.w)/2.+1e-4:self.dx]
        x = x.ravel()
        y = y.ravel()

        dx = self.dx
        d = self.dw0
        hdx = self.hdx

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        pa = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h=h, h0=h0,
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
            output_at_times=[0.1, 0.2, 0.4, 0.6],
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(dest='fluid',
                                                           sources=['fluid'])
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
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid'])
                    ]
                ),
            Group(
                equations=[
                    DaughterVelocityEval(rhow=rho_w, dest='fluid',
                                         sources=['fluid'])
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid')
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim=dim, dest='fluid',
                                         sources=['fluid'], u_only=True)
                    ]
                ),
            Group(
                equations=[
                   FindMergeable(dest='fluid', sources=['fluid'],
                                 A_min=self.A_min, x_min=self.x_min_merge)
                   ], update_nnps=True
                 ),
            Group(
                equations=[
                        InitialDensityEvalAfterMerge(dest='fluid',
                                                     sources=['fluid'])
                    ]
                 ),
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(dest='fluid',
                                                           sources=['fluid'])
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
                    )], iterate=True, max_iterations=100
            ),
            Group(
                equations=[
                    CheckForParticlesToSplit(
                        dest='fluid', h_max=self.h_max, A_max=self.A_max,
                        x_min=self.x_min_split, x_max=self.x_max_split)
                    ],
                ),
            ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.nnps.update()

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        import matplotlib
        matplotlib.use('Agg')
        self._plot_depth_of_dam()

    def _plot_depth_of_dam(self):
        from matplotlib import pyplot as plt
        t_to_plot = [0.1, 0.2, 0.4, 0.6]
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
            plt.plot(x_actual, dw_actual, 'r-', label='Analytical Solution')
            plt.plot(x, dw, 'bo ', label='SWE-SPH', markersize=2)
            plt.ylim(0, 1.2*self.dw0)
            plt.xlim(0, 1.5*x_max)
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('dw (m)')
            plt.title('t = %0.1fs' % t_to_plot[num])
            fig = os.path.join(self.output_dir, "depth%d.png" % (num+1))
            plt.savefig(fig, dpi=300)
            num += 1


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        SWEOS(dest='fluid')
                        ]
                    ),
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreakSplitandMerge()
    app.run()
    app.post_process(app.info_filename)
