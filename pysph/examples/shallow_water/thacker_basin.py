""" Thacker Basin Test Case (Oscillation of a fluid column in a frictionless
paraboloid basin) (4.4 hr)

The case is described in "Accurate particle splitting for smoothed particle
hydrodynamics in shallow water with shock capturing", R. Vacondio, B. D. Rogers
and P.K. Stansby, Int. J. Numer. Meth. Fluids, Vol 69, pp 1377-1410 (2012).
DOI: 10.1002/fld.2646

"""
import os

# Numpy
from numpy import (ones_like, zeros, zeros_like, mgrid, pi, array,
                   arange, sqrt, concatenate, sin, cos, savez)

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

# PySPH Interpolator
from pysph.tools.interpolator import Interpolator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2.0


class ThackerBasin(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.5,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--r", action="store", type=float, dest="r", default=3000.0,
            help="Initial radius of the fluid column (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=75,
            help="Number of concentric fluid particle circles (Determines\
            spacing btw particles, dr = r/n)")
        group.add_argument(
            "--x_cen_fluid", action="store", type=float, dest="x_cen_fluid",
            default=1500.0, help="x-coordinate of center of the circular fluid\
            column (m)")
        group.add_argument(
            "--y_cen_fluid", action="store", type=float, dest="y_cen_fluid",
            default=0.0, help="y-coordinate of center of the circular fluid\
            column (m)")
        group.add_argument(
            "--zo", action="store", type=float, dest="zo", default=10.0,
            help="Distance of parabola origin from bottom (m)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.r = self.options.r
        self.n = self.options.n
        self.x_cen_fluid = self.options.x_cen_fluid
        self.y_cen_fluid = self.options.y_cen_fluid
        self.zo = self.options.zo
        self.omega = sqrt(2*g*self.zo) / self.r  # Angular Frequency

    def create_particles(self):
        n = self.n
        hdx = self.hdx
        fluid_rad = self.r
        dr = (fluid_rad-100) / n
        zo = self.zo

        # Bed
        dxb = 50.
        xb, yb = mgrid[-5000:5000:dxb, -5000:5000:dxb]
        b = zo * ((xb**2 + yb**2)/fluid_rad**2)

        Vb = ones_like(xb) * dxb * dxb
        hb = ones_like(xb) * hdx * dxb

        bed = gpa_swe(name='bed', x=xb, y=yb, V=Vb, b=b, h=hb)

        # For gradient correction
        len_b = len(bed.x) * 9
        bed.add_constant('m_mat', [0.0] * len_b)

        # Fluid
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

        x += self.x_cen_fluid
        y += self.y_cen_fluid

        h = ones_like(x) * hdx * dr
        h0 = ones_like(x) * hdx * dr

        # Distance between fluid center and bed center along x-direction
        zeta = self.x_cen_fluid - 0.0

        u = zeros_like(x)
        v = ones_like(x) * -(zeta*self.omega)
        vh = ones_like(x) * -(zeta*self.omega)

        # At t = 0.0s
        fluid_surf_hei = zo + (2*zeta*(zo/fluid_rad)
                               * ((x/fluid_rad) - (zeta/(2.0*fluid_rad))))

        fluid = gpa_swe(x=x, y=y, h=h, h0=h0, u=u, v=v, vh=vh, name='fluid')

        compute_fluid_elevation([fluid, bed])

        dw = fluid_surf_hei - fluid.b

        rho = dw * rho_w
        rho0 = dw * rho_w
        m = rho * (1.56*dr*dr)

        fluid.m = m
        fluid.rho = rho
        fluid.rho0 = rho0
        fluid.dw = dw

        compute_initial_props([fluid])

        return [fluid, bed]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = (2*pi) / self.omega
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=[0.015*tf, 0.505*tf, 0.980*tf],
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
                    SWEOS(dest='fluid'),
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
                    ParticleAcceleration(dim=dim, dest='fluid',
                                         sources=['fluid'])
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
        self._plot_depth_and_vel()

    def _plot_depth_and_vel(self):
        from matplotlib import pyplot as plt
        t_arr = []
        u_arr = []
        dw_arr = []
        v_arr = []

        # Properties of the fluid at this location are found at various times
        # Note: This location is wetted at all times by the fluid
        x_loc_to_interpolate = 0.0
        y_loc_to_interpolate = 0.0

        kernel = CubicSpline(dim=2)
        for fname in self.output_files:
            data = load(fname)
            fluid = data['arrays']['fluid']
            t_arr.append(data['solver_data']['t'])
            interp = Interpolator([fluid], kernel=kernel)
            interp.set_interpolation_points(x_loc_to_interpolate,
                                            y_loc_to_interpolate)
            u_interp = interp.interpolate('u')
            v_interp = interp.interpolate('v')
            dw_interp = interp.interpolate('dw')
            u_arr.append(u_interp.item())
            v_arr.append(v_interp.item())
            dw_arr.append(dw_interp.item())

        t = array(t_arr)
        x = zeros_like(t)
        y = zeros_like(t)
        fluid_rad = self.r
        zo = self.zo
        zeta = self.x_cen_fluid - 0.0
        omega = self.omega

        def actual_fluid_surf_hei(x, y, t):
            fluid_surf_hei = zo + (2*zeta*(zo/fluid_rad)
                                   * ((x/fluid_rad)*cos(omega*t)
                                   - (y/fluid_rad)*sin(omega*t)
                                   - (zeta/(2.0*fluid_rad))))
            return fluid_surf_hei

        def fluid_bottom_hei(x, y):
            return zo * ((x**2+y**2)/fluid_rad**2)

        def u_actual_func(t):
            return -zeta*omega*sin(omega*t)

        def v_actual_func(t):
            return -zeta*omega*cos(omega*t)

        dw_actual = actual_fluid_surf_hei(x, y, t) - fluid_bottom_hei(x, y)
        u_actual = u_actual_func(t)
        v_actual = v_actual_func(t)

        # Time period
        T = (2*pi) / omega

        if len(dw_arr) == 0 or len(u_arr) == 0 or len(v_arr) == 0:
            return

        dw_min = min(dw_arr)
        dw_max = max(dw_arr)
        u_min = min(u_arr)
        u_max = max(u_arr)
        v_min = min(v_arr)
        v_max = max(v_arr)

        b = fluid_bottom_hei(x, y)

        fname_res = os.path.join(self.output_dir, 'results.npz')
        savez(
            fname_res, t=t_arr, dw_num=dw_arr, dw_actual=dw_actual,
            u_num=u_arr, u_actual=u_actual, v_num=v_arr, v_actual=v_actual,
            x=x, y=y, b=b
            )

        plt.clf()
        plt.plot(t_arr/T, dw_actual, 'r-', label='Analytical Solution')
        plt.plot(t_arr/T, dw_arr, 'bo', markersize=2, label='SWE-SPH')
        plt.legend()
        plt.ylim(0.8*dw_min, 1.5*dw_max)
        plt.xlim(0, 1)
        plt.xlabel('t/T')
        plt.ylabel('dw (m)')
        fig = os.path.join(self.output_dir, "depth")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t_arr/T, u_actual, 'ro', markersize=2,
                 label='Analytical Solution')
        plt.plot(t_arr/T, u_arr, 'bo', markersize=2, label='SWE-SPH')
        plt.legend()
        plt.ylim(1.5*u_min, 1.5*u_max)
        plt.xlim(0, 1)
        plt.xlabel('t/T')
        plt.ylabel('u (m/s)')
        fig = os.path.join(self.output_dir, "velocity1")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t_arr/T, v_actual, 'ro', markersize=2,
                 label='Analytical Solution')
        plt.plot(t_arr/T, v_arr, 'bo', markersize=2, label='SWE-SPH')
        plt.legend()
        plt.ylim(1.5*v_min, 1.5*v_max)
        plt.xlim(0, 1)
        plt.xlabel('t/T')
        plt.ylabel('v (m/s)')
        fig = os.path.join(self.output_dir, "velocity2")
        plt.savefig(fig, dpi=300)


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
                    BedGradient(dest='bed', sources=['bed']),
                    ]
                ),
            Group(
                equations=[
                    BedCurvature(dest='bed', sources=['bed']),
                    ]
                ),
        ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = ThackerBasin()
    app.run()
    app.post_process(app.info_filename)
