"""Cylindrical dam break over a dry bed. (1.5 mins)

The case is described in "A corrected smooth particle hydrodynamics formulation
of the shallow-water equations", Miguel Rodriguez-Paz and Javier Bonet,
Computers & Structures, Vol 83, pp 1396-1410 (2005).
DOI:10.1016/j.compstruc.2004.11.025

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


class CylindricalDamBreak(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=1.0,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--r", action="store", type=float, dest="r", default=0.5,
            help="Initial radius of the fluid column (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=50,
            help="Number of concentric fluid particle circles (Determines\
            spacing btw particles, dr = r/n)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dw0 = self.options.dw0
        self.r = self.options.r
        self.n = self.options.n

    def create_particles(self):
        n = self.n
        r = self.r
        dr = r / n

        d = self.dw0
        hdx = self.hdx

        x = zeros(0)
        y = zeros(0)

        # Create circular patch of fluid in a radial grid
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
        tf = 1.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=(0.1, 0.2, 0.3),
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
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid']),
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
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid']),
                    SWEOS(dest='fluid')
                ]
            ),

            Group(
                equations=[
                    ParticleAcceleration(dim=dim, dest='fluid',
                                         sources=['fluid'])
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
        t_to_plot = [0.1, 0.2, 0.3]
        fname_for_plot = []
        for fname in self.output_files:
            data = load(fname)
            t = data['solver_data']['t']
            if t in t_to_plot:
                fname_for_plot.append(fname)

        num = 0
        # Solution files of this problem (Rodriguez et al.) for
        # depth of dam 1 m and radius 0.5 m at times 0.1s, 0.2s & 0.3s
        files_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'files_for_output_comparison'
                )
        rodri_soln_files = [
                os.path.join(files_dir, 'cyl_dam_t01.csv'),
                os.path.join(files_dir, 'cyl_dam_t02.csv'),
                os.path.join(files_dir, 'cyl_dam_t03.csv')
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
            if (self.dw0 == 1.0 and self.r == 0.5 and (t in [0.1, 0.2, 0.3])):
                file_num = [0.1, 0.2, 0.3].index(t)
                x_rodri, dw_rodri = loadtxt(
                        rodri_soln_files[file_num], delimiter=',', unpack=True)
                x_rodri_idx_sort = x_rodri.argsort()
                x_rodri = x_rodri[x_rodri_idx_sort]
                dw_rodri = dw_rodri[x_rodri_idx_sort]
                plt.plot(x_rodri, dw_rodri, 'r-', label='Rodriguez Solution')

                fname_res = os.path.join(
                    self.output_dir, 'results%d.npz' % (num+1))
                savez(
                    fname_res, t=t_to_plot[num], x=x, x_rodri=x_rodri, y=y,
                    dw=dw, dw_rodri=dw_rodri, u=u, v=v, rho=rho, au=au, av=av)

            plt.plot(x, dw, 'bo', label='SWE-SPH', markersize=2)
            plt.ylim(0, 1.2*self.dw0)
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
    app = CylindricalDamBreak()
    app.run()
    app.post_process(app.info_filename)
