"""1D Rectangular dam break over a wet bed. (5 sec)

The case is described in "Accurate particle splitting for smoothed particle
hydrodynamics in shallow water with shock capturing", R. Vacondio, B. D. Rogers
and P. K. Stansby, Int. J. Numer. Meth. Fluids, Vol 69, pp 1377-1410 (2012).
DOI: 10.1002/fld.2646

"""
import os

# Numpy
from numpy import (ones_like, array, arange, sqrt, concatenate, where, savez)

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
        ParticleAcceleration
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 1


class RectangularDamBreak(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx1", action="store", type=float, dest="dx1", default=5.0,
            help="Spacing between the particles of first fluid column")
        group.add_argument(
            "--dx2", action="store", type=float, dest="dx2", default=10.0,
            help="Spacing between the particles of second fluid column")
        group.add_argument(
            "--dw01", action="store", type=float, dest="dw01", default=10.0,
            help="Initial depth of the first fluid column (m)")
        group.add_argument(
            "--dw02", action="store", type=float, dest="dw02", default=5.0,
            help="Initial depth of the second fluid column (m)")
        group.add_argument(
            "--l1", action="store", type=float, dest="l1", default=1000.0,
            help="Initial length of the first fluid column (m)")
        group.add_argument(
            "--l2", action="store", type=float, dest="l2", default=1000.0,
            help="Initial length of the second fluid column (m)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx1 = self.options.dx1
        self.dx2 = self.options.dx2
        self.dw01 = self.options.dw01
        self.dw02 = self.options.dw02
        self.l1 = self.options.l1
        self.l2 = self.options.l2

    def create_particles(self):
        hdx = self.hdx
        dx1 = self.dx1
        dx2 = self.dx2
        l1 = self.l1
        l2 = self.l2
        tot_l = l1 + l2

        d1 = self.dw01
        d2 = self.dw02

        x = concatenate(
                (arange(0, l1, dx1), arange(l1, tot_l+1e-4, dx2)), axis=0
                )

        m = ones_like(x)
        h = ones_like(x)
        h0 = ones_like(x)
        rho = ones_like(x)
        rho0 = ones_like(x)

        # Setting first fluid column properties
        idx_fluid_col_1 = where(x < 1000.0)[0]
        m[idx_fluid_col_1] *= dx1 * rho_w * d1
        h[idx_fluid_col_1] *= hdx * dx1
        h0[idx_fluid_col_1] *= hdx * dx1
        rho[idx_fluid_col_1] *= rho_w * d1
        rho0[idx_fluid_col_1] *= rho_w * d1

        # Setting second fluid column properties
        idx_fluid_col_2 = where(x >= 1000.0)[0]
        m[idx_fluid_col_2] *= dx2 * rho_w * d2
        h[idx_fluid_col_2] *= hdx * dx2
        h0[idx_fluid_col_2] *= hdx * dx2
        rho[idx_fluid_col_2] *= rho_w * d2
        rho0[idx_fluid_col_2] *= rho_w * d2

        fluid = gpa_swe(x=x, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                        name='fluid')

        # Closed Boundary
        x = concatenate(
                (arange(-2*dx1, l1, dx1), arange(l1, tot_l+2*dx2+1e-4, dx2)),
                axis=0)
        idx_cb_left = where((x < 0))[0]
        idx_cb_right = where((x > 2000))[0]

        m_cb = ones_like(x)
        h_cb = ones_like(x)
        rho_cb = ones_like(x)
        dw_cb = ones_like(x)
        cs_cb = ones_like(x)
        alpha_cb = ones_like(x)

        m_cb[idx_cb_left] *= dx1 * rho_w * d1
        h_cb[idx_cb_left] *= hdx * dx1
        rho_cb[idx_cb_left] *= rho_w * d1
        dw_cb[idx_cb_left] *= d1
        cs_cb[idx_cb_left] *= sqrt(9.8 * dw_cb[idx_cb_left])
        alpha_cb[idx_cb_left] *= dim * rho_cb[idx_cb_left]

        m_cb[idx_cb_right] *= dx2 * rho_w * d2
        h_cb[idx_cb_right] *= hdx * dx2
        rho_cb[idx_cb_right] *= rho_w * d2
        dw_cb[idx_cb_right] *= d2
        cs_cb[idx_cb_right] *= sqrt(9.8 * dw_cb[idx_cb_right])
        alpha_cb[idx_cb_right] *= dim * rho_cb[idx_cb_right]

        boundary = gpa_swe(name='boundary', x=x, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb)

        idx_to_remove = where((x >= 0) & (x <= 2000))[0]
        boundary.remove_particles(idx_to_remove)

        compute_initial_props([fluid, boundary])
        return [fluid, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=1)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 60
        solver = Solver(
            kernel=kernel,
            dim=1,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=[10, 20, 30, 40, 50, 60],
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(
                                dest='fluid', sources=['fluid', 'boundary']
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
                        dest='fluid', sources=['fluid', 'boundary'])
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid')
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(
                        dim=dim, dest='fluid', sources=['fluid', 'boundary'],
                        visc_option=2, u_only=True)
                    ],
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
        t_to_plot = [50.0]
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

            cond1 = fluid.x >= 0
            cond2 = fluid.x <= 2000
            cond = cond1 & cond2
            x = fluid.x[cond]
            dw = fluid.dw[cond]

            x_ind_sort = x.argsort()
            x = x[x_ind_sort]
            x_max = max(x)
            dw = dw[x_ind_sort]
            dw_max = max(self.dw01, self.dw02)
            dw_min = min(self.dw01, self.dw02)

            plt.clf()
            if (
                    (self.dw01 == 10. and self.dw02 == 5.)
                    and (self.l1 == 1000. and self.l2 == 1000.)
                    and abs(t-50.0) < 1e-10
            ):
                x_actual = array(
                        [0., 500., 721.87499, 1468.750, 1468.750, 2000.])
                dw_actual = array([10., 10., 7.27083, 7.27083, 5., 5.])
                plt.plot(x_actual, dw_actual, 'r-', label='Exact Solution',
                         linewidth=2.5)
                fname_res = os.path.join(
                    self.output_dir, 'results.npz')
                savez(
                    fname_res, t=t_to_plot[num], x=x, x_actual=x_actual,
                    dw=dw, dw_actual=dw_actual)

            plt.plot(x, dw, 'bo', label='SWE-SPH', markersize=2)
            plt.ylim(0.8*dw_min, 1.2*dw_max)
            plt.xlim(0, x_max+self.dx2)

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
                SWEOS(dest='fluid')
                ],
            )
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreak()
    app.run()
    app.post_process(app.info_filename)
