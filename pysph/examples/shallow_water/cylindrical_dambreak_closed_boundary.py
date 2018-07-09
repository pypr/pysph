"""Cylindrical dam break over a dry bed enclosed in a circular closed boundary.
(12 min)

The case is described in "Smoothed Particle Hydrodynamics: Approximate
zero-consistent 2-D boundary conditions and still shallow-water tests",
R. Vacondio, B.D. Rodgers and P.K. Stansby, Int. J. Numer. Meth. Fluids,
69 (2012), pp. 226-253. DOI: 10.1002/fld.2559

"""
import os

# Numpy
from numpy import (ones_like, zeros, mgrid, pi, arange, sqrt, concatenate,
                   sin, cos, where, intersect1d, loadtxt, savez)

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
dim = 2


class CylindricalDamBreakClosedBoundary(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=10.0,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--r", action="store", type=float, dest="r", default=10.0,
            help="Initial radius of the fluid column (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=105,
            help="Number of concentric fluid particle circles (Determines\
            spacing btw particles, dr = r/n)")
        group.add_argument(
            "--inner_r_wall", action="store", type=float, dest="inner_r_wall",
            default=15.0, help="Inner radius of the circular wall (m)")
        group.add_argument(
            "--n_wall", action="store", type=float, dest="n_wall", default=10,
            help="Number of concentric wall boundary particle circles")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dw0 = self.options.dw0
        self.r = self.options.r
        self.n = self.options.n
        self.inner_r_wall = self.options.inner_r_wall
        self.n_wall = self.options.n_wall

    def create_particles(self):
        # Fluid
        n = self.n
        r = self.r
        dr = r / n

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

        fluid = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                        name='fluid')

        compute_initial_props([fluid])

        # Circular Closed Boundary
        inner_r_wall = self.inner_r_wall
        x, y = mgrid[-1.5*inner_r_wall:1.5*inner_r_wall:dr,
                     -1.5*inner_r_wall:1.5*inner_r_wall:dr]
        x = x.ravel()
        y = y.ravel()
        idx1 = where(inner_r_wall**2 <= (x**2+y**2))[0]
        idx2 = where((x**2+y**2) < (inner_r_wall+self.n_wall*dr)**2)
        idx = intersect1d(idx1, idx2)
        x_cb, y_cb = x[idx], y[idx]

        m_cb = ones_like(x_cb) * (1.56*dr*dr) * rho_w * d
        h_cb = ones_like(x_cb) * hdx * dr
        rho_cb = ones_like(x_cb) * rho_w * d
        dw_cb = ones_like(x_cb) * d
        cs_cb = sqrt(9.8 * dw_cb)
        alpha_cb = dim * rho_cb

        # Tags wall boundary particles to set virtual depth and ignore
        # artificial viscosity interaction
        is_wall_boun_pa = ones_like(x_cb)

        boundary = gpa_swe(name='boundary', x=x_cb, y=y_cb, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb, is_wall_boun_pa=is_wall_boun_pa)

        return [fluid, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 2.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.1,
            adaptive_timestep=True,
            output_at_times=(0.1, 0.4, 2.0),
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
                        dest='fluid', sources=['fluid', 'boundary']),
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
                                         sources=['fluid', 'boundary']),
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
        t_to_plot = [0.1, 0.4, 2.0]
        fname_for_plot = []
        for fname in self.output_files:
            data = load(fname)
            t = data['solver_data']['t']
            if t in t_to_plot:
                fname_for_plot.append(fname)

        num = 0
        # Solution files of this problem (Vacondio et al.) for
        # depth of dam 10 m and radius 10.0 m at times 0.1s, 0.4s & 2.0s
        files_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'files_for_output_comparison'
                )
        vacondio_soln_files = [
                os.path.join(files_dir, 'cyl_dam_closed_boun_t01.csv'),
                os.path.join(files_dir, 'cyl_dam_closed_boun_t02.csv'),
                os.path.join(files_dir, 'cyl_dam_closed_boun_t03.csv')
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
                self.r == 10.0 and
                (t in [0.1, 0.4, 2.0])
            ):
                file_num = [0.1, 0.4, 2.0].index(t)
                x_vacondio, dw_vacondio = loadtxt(
                        vacondio_soln_files[file_num], delimiter=',',
                        unpack=True)
                x_vacondio_idx_sort = x_vacondio.argsort()
                x_cen_vacondio = 15.0
                x_vacondio = x_vacondio[x_vacondio_idx_sort] - x_cen_vacondio
                dw_vacondio = dw_vacondio[x_vacondio_idx_sort]
                plt.plot(x_vacondio, dw_vacondio, 'r-',
                         label='Vacondio Solution')

                fname_res = os.path.join(
                    self.output_dir, 'results%d.npz' % (num+1))
                savez(
                    fname_res, t=t_to_plot[num], x=x, x_vacondio=x_vacondio,
                    y=y, dw=dw, dw_vacondio=dw_vacondio, u=u, v=v, rho=rho,
                    au=au, av=av)

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
                SWEOS(dest='fluid'),
                ],
            )
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = CylindricalDamBreakClosedBoundary()
    app.run()
    app.post_process(app.info_filename)
