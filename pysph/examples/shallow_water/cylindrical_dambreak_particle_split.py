"""Cylindrical dam break over a dry bed with particle splitting. (67 min)

The case is described in "Accurate particle splitting for smoothed particle
hydrodynamics in shallow water with shock capturing", R. Vacondio, B. D. Rogers
and P.K. Stansby, Int. J. Numer. Meth. Fluids, 69 (2012), pp. 1377-1410.
DOI: 10.1002/fld.2646

"""
import os

# Numpy
from numpy import (ones_like, zeros, pi, arange, concatenate, sin, cos,
                   loadtxt, savez)

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
        DaughterVelocityEval
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class CylindricalDamBreakSplit(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=10.0,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--r", action="store", type=float, dest="r", default=500.0,
            help="Initial radius of the fluid column (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=50,
            help="Number of concentric fluid particle circles (Determines\
            spacing btw particles, dr = r/n)")
        group.add_argument(
            "--coeff_A_split", action="store", type=float,
            dest="coeff_A_max", default=2.0, help="Ratio of area of\
            particle to initial area after which the particle splitting is\
            activated")
        group.add_argument(
            "--coeff_h_split", action="store", type=float, dest="coeff_h_max",
            default=3.0, help="Ratio of smoothing length of particle to\
            initial smoothing length after which the particle splitting is\
            deactivated")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dw0 = self.options.dw0
        self.r = self.options.r
        self.n = self.options.n
        self.coeff_A_max = self.options.coeff_A_max
        self.coeff_h_max = self.options.coeff_h_max
        self.dr = self.r / self.n

        # Particle splitting activated when A_i > A_max and h_i < h_max
        self.A_max = self.coeff_A_max * (1.56*self.dr**2)
        self.h_max = self.coeff_h_max * self.hdx * self.dr

    def create_particles(self):
        dr = self.dr
        n = self.n
        d = self.dw0
        hdx = self.hdx

        x = zeros(0)
        y = zeros(0)

        # Create circular patch in a radial grid
        rad = 0.0
        for j in range(1, n+1):
                npnts = 4 * j
                dtheta = (2*pi) / npnts

                theta = arange(0, 2*pi-1e-10, dtheta)
                rad = rad + dr

                _x = rad * cos(theta)
                _y = rad * sin(theta)

                x = concatenate((x, _x))
                y = concatenate((y, _y))

        m = ones_like(x) * (1.56*dr*dr) * rho_w * d

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        h = ones_like(x) * hdx * dr
        h0 = ones_like(x) * hdx * dr

        pa = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                     name='fluid')

        compute_initial_props([pa])
        return [pa]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 50.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=(10.0, 30.0, 50.0),
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
                        dest='fluid', sources=['fluid']),
                    ]
                ),
            Group(
                equations=[
                    DaughterVelocityEval(rhow=rho_w, dest='fluid',
                                         sources=['fluid']),
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim=dim, dest='fluid',
                                         sources=['fluid']),
                    ]
                ),
            Group(
                equations=[
                    CheckForParticlesToSplit(dest='fluid', h_max=self.h_max,
                                             A_max=self.A_max)
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
        t_to_plot = [10.0, 30.0, 50.0]
        fname_for_plot = []
        for fname in self.output_files:
            data = load(fname)
            t = data['solver_data']['t']
            if t in t_to_plot:
                fname_for_plot.append(fname)

        num = 0
        # Finite volume solution files of this problem for depth of dam 10 m
        # and radius 1000 m at times 10s, 30s and 50s
        files_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'files_for_output_comparison'
                )
        finite_vol_soln_files = [
                os.path.join(files_dir, 'cyl_dam_split_t01.csv'),
                os.path.join(files_dir, 'cyl_dam_split_t02.csv'),
                os.path.join(files_dir, 'cyl_dam_split_t03.csv')
                ]
        for fname in fname_for_plot:
            data = load(fname)
            fluid = data['arrays']['fluid']
            t = data['solver_data']['t']

            cond = abs(fluid.y) < 1e-2
            x = fluid.x[cond]
            y = fluid.y[cond]

            if len(x) == 0:
                return

            x_max = max(x)
            dw = fluid.dw[cond]
            rho = fluid.rho[cond]
            u = fluid.u[cond]
            v = fluid.v[cond]
            au = fluid.au[cond]
            av = fluid.av[cond]

            plt.clf()
            if (
                self.dw0 == 10.0 and
                self.r == 500.0 and
                (t in [10.0, 30.0, 50.0])
            ):
                file_num = [10.0, 30.0, 50.0].index(t)
                x_fv, dw_fv = loadtxt(
                    finite_vol_soln_files[file_num], delimiter=',',
                    unpack=True)
                x_fv_idx_sort = x_fv.argsort()
                x_cen_fv = 1500.0
                x_fv = x_fv[x_fv_idx_sort] - x_cen_fv
                dw_fv = dw_fv[x_fv_idx_sort]
                plt.plot(x_fv, dw_fv, 'r-', label='Finite Volume Solution')

                fname_res = os.path.join(
                    self.output_dir, 'results%d.npz' % (num+1))
                savez(
                    fname_res, t=t_to_plot[num], x=x, x_fv=x_fv, y=y,
                    dw=dw, dw_fv=dw_fv, u=u, v=v, rho=rho, au=au, av=av)

            plt.plot(x, dw, 'bo', label='SWE-SPH', markersize=2)
            plt.ylim(0, 1.5*self.dw0)
            plt.xlim(-1.5*x_max, 1.5*x_max)
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
                SWEOS(dest='fluid'),
                ],
            )
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = CylindricalDamBreakSplit()
    app.run()
    app.post_process(app.info_filename)
