"""" Okushiri Tsunami Test Case (101 hr)

The numerical case is described in "Shallow Water and Navier-Stokes SPH-like
numerical modelling of rapidly varying free-surface flows", Renato Vacondio,
Doctoral dissertation, University of Parma, pp 104-112 (2010).

The experimental data can be obtained from the below url
http://isec.nacse.org/workshop/2004_cornell/bmark2.html

"""
import os

# Numpy
from numpy import (ones_like, zeros, zeros_like, mgrid, ravel, loadtxt, arange,
                   sqrt, concatenate, where, array, savez)

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
        BedGradient, BedCurvature, SWEInlet, SWEInletOutletStep,
        RemoveFluidParticlesWithNoNeighbors, RemoveParticlesWithZeroAlpha,
        RemoveCloseParticlesAtOpenBoundary, BoundaryInnerReimannStateEval,
        SubCriticalTimeVaryingOutFlow, BedFrictionSourceEval,
        RemoveOutofDomainParticles
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator

# PySPH Interpolator
from pysph.tools.interpolator import Interpolator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class OkushiriTsunami(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.01875,
            help="Spacing between the particles")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=0.13535,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--le", action="store", type=float, dest="le", default=5.448,
            help="Length of the fluid domain (m)")
        group.add_argument(
            "--w", action="store", type=float, dest="w", default=3.402,
            help="Width of the fluid domain (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=0.025,
            help="Manning coefficient ")
        group.add_argument(
            "--Vb", action="store", type=float, dest="Vb", default=1.96e-4,
            help="Volume of the bed particles")
        group.add_argument(
            "--hb", action="store", type=float, dest="hb", default=1.68e-2,
            help="Smoothing length of the bed particles")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.dw0 = self.options.dw0
        self.le = self.options.le
        self.w = self.options.w
        self.n = self.options.n
        self.Vb = self.options.Vb
        self.hb = self.options.hb

        # Inlet
        self.num_inlet_pa = 2
        self.x_max_inlet = 0
        self.x_min_inlet = -0.95 * self.dx * self.num_inlet_pa
        # Minimum distance allowed between open boundary particles
        self.min_dist_ob = self.dx / 2.

    def initialize(self):
        # Directory containing input files
        self.dir_input_files = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'okushiri_tsunami_input_files')
        # File containing open boundary conditions
        obc_fname = os.path.join(self.dir_input_files, 'tsunami_obc.txt')
        # Loads the time varying depth field at inlet
        self.t_ob, self.dw_ob = loadtxt(obc_fname, delimiter=' ', unpack=True)

    def create_particles(self):
        hdx = self.hdx
        dx = self.dx
        d = self.dw0
        w = self.w
        y_max = w
        l_domain = self.le

        # Inlet Properties
        y = arange(dx/2, w-(dx/4.), dx)
        x = zeros_like(y) - 0.5*dx

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        # Stores the time-varying depth field imposed at inlet
        dw_at_t = ones_like(x) * d

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d
        alpha = dim * rho
        cs = sqrt(9.8 * rho/rho_w)

        inlet = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h0=h0, h=h,
                        dw_at_t=dw_at_t, alpha=alpha, cs=cs, name='inlet')
        boundary_props = ['dw_inner_reimann', 'u_inner_reimann',
                          'v_inner_reimann', 'shep_corr']
        inlet.add_output_arrays(boundary_props)

        # Bed Properties
        bed_fname = os.path.join(self.dir_input_files, 'tsunami_bed.txt.bz2')
        xb, yb, b = loadtxt(bed_fname, delimiter=' ', unpack=True)

        Vb = ones_like(xb) * self.Vb
        nb = ones_like(xb) * self.n  # Manning Coefficient
        hb = ones_like(xb) * self.hb

        bed = gpa_swe(name='bed', x=xb, y=yb, V=Vb, n=nb, h=hb, b=b)
        len_b = len(bed.x) * 9
        bed.add_constant('m_mat', [0.0] * len_b)

        # Fluid Properties
        xf, yf = mgrid[0.5*dx:self.x_max_inlet+l_domain:dx,
                       dx/2:y_max-(dx/4.):dx]
        xf, yf = (ravel(t) for t in (xf, yf))
        h = ones_like(xf) * hdx * dx
        h0 = ones_like(xf) * hdx * dx
        fluid = gpa_swe(name='fluid', x=xf, y=yf, h=h, h0=h0)

        compute_fluid_elevation([fluid, bed])

        fluid_surf_hei = d
        dw = fluid_surf_hei - fluid.b
        rho = dw * rho_w
        rho0 = dw * rho_w
        m = rho * dx**2

        fluid.m = m
        fluid.rho = rho
        fluid.rho0 = rho0
        fluid.dw = dw

        # Removes fluid particles with depth less than d_min
        d_min = 7e-5
        idx = where(fluid.dw < d_min)[0]
        fluid.remove_particles(idx)

        # Closed Boundary
        # Note: 5 layers of wall boundary particles
        xcb_top = arange(self.x_min_inlet-5*dx, l_domain+5*dx, dx/2.)
        ycb_top = zeros(0)
        for i in arange(-0.5, 2, 0.5):
            ycb_top = concatenate((ycb_top, ones_like(xcb_top)*(y_max+i*dx)),
                                  axis=0)

        xcb_top = concatenate((xcb_top, xcb_top+dx/4., xcb_top,
                               xcb_top+dx/4., xcb_top), axis=0)

        xcb_bottom = arange(self.x_min_inlet-5*dx, l_domain+5*dx, dx/2.)
        ycb_bottom = zeros(0)
        for i in arange(0, -2.5, -0.5):
            ycb_bottom = concatenate((ycb_bottom, zeros_like(xcb_bottom)+i*dx),
                                     axis=0)

        xcb_bottom = concatenate((xcb_bottom, xcb_bottom+dx/4., xcb_bottom,
                                  xcb_bottom+dx/4., xcb_bottom), axis=0)

        ycb_right = arange(dx/4., y_max-(dx/4.), dx/2.)
        xcb_right = zeros(0)
        for i in arange(0.5, 3.0, 0.5):
            xcb_right = concatenate((xcb_right, zeros_like(ycb_right)
                                     + (l_domain+i*dx)), axis=0)

        ycb_right = concatenate((ycb_right, ycb_right+dx/4., ycb_right,
                                 ycb_right+dx/4, ycb_right), axis=0)

        xcb_all = concatenate((xcb_top, xcb_bottom, xcb_right), axis=0)
        ycb_all = concatenate((ycb_top, ycb_bottom, ycb_right), axis=0)

        m_cb = ones_like(xcb_all) * dx/2. * dx/2. * rho_w * d
        h_cb = ones_like(xcb_all) * hdx * dx/2.
        rho_cb = ones_like(xcb_all) * rho_w * d
        dw_cb = ones_like(xcb_all) * d
        cs_cb = sqrt(9.8 * dw_cb)
        alpha_cb = dim * rho_cb
        is_wall_boun_pa = ones_like(xcb_all)

        boundary = gpa_swe(name='boundary', x=xcb_all, y=ycb_all, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           is_wall_boun_pa=is_wall_boun_pa, alpha=alpha_cb)

        return [inlet, fluid, bed, boundary]

    def create_inlet_outlet(self, particle_arrays):
        f_pa = particle_arrays['fluid']
        i_pa = particle_arrays['inlet']
        b_pa = particle_arrays['bed']
        cb_pa = particle_arrays['boundary']

        inlet = SWEInlet(
            i_pa, f_pa, f_pa, spacing=self.dx, n=self.num_inlet_pa, axis='x',
            xmin=self.x_min_inlet, xmax=self.x_max_inlet, ymin=0, ymax=self.w
        )

        compute_initial_props([i_pa, f_pa, b_pa, cb_pa])

        return [inlet]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(inlet=SWEInletOutletStep(), fluid=SWEStep())
        tf = 22.51
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.4,
            adaptive_timestep=True,
            output_at_times=(10, 12, 14, 15, 16, 17, 18, 20),
            tf=tf
            )
        return solver

    def pre_step(self, solver):
        t = solver.t
        # Sets the time varying depth for inlet particles based on the current
        # time t
        for pa in self.particles:
            if pa.name == 'inlet':
                for i in range(len(self.t_ob)-1):
                    if t < self.t_ob[i+1]:
                        # Calculates the slope of depth vs time
                        m1 = ((self.dw_ob[i+1]-self.dw_ob[i])
                              / (self.t_ob[i+1]-self.t_ob[i]))
                        # Interpolates the depth based on current time and sets
                        # it for the inlet particles
                        pa.dw_at_t = (ones_like(pa.x)
                                      * (self.dw_ob[i]+m1*(t-self.t_ob[i])))
                        break

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    RemoveFluidParticlesWithNoNeighbors(
                        dest='fluid', sources=['fluid'])
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    RemoveOutofDomainParticles(
                        dest='fluid', x_min=self.x_max_inlet, x_max=self.le,
                        y_min=0, y_max=self.w
                        )
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    RemoveCloseParticlesAtOpenBoundary(
                       min_dist_ob=self.min_dist_ob, dest='inlet',
                       sources=['inlet']
                       )
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(
                                dest='fluid',
                                sources=['inlet', 'fluid', 'boundary']
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
                        dest='fluid', sources=['fluid', 'inlet', 'boundary']),
                        ],
                ),
            Group(
                equations=[
                   RemoveParticlesWithZeroAlpha(dest='fluid'),
                        ], update_nnps=True
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    BoundaryInnerReimannStateEval(dest='inlet',
                                                  sources=['fluid']),
                    ]
                ),
            Group(
                equations=[
                    SubCriticalTimeVaryingOutFlow(dest='inlet'),
                    ]
                ),
            Group(
                equations=[
                    BedFrictionSourceEval(
                        dest='fluid', sources=['bed'])
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
                        dim=dim, dest='fluid',
                        sources=['fluid', 'inlet', 'boundary'],
                        ),
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
        self._plot_depth()

    def _plot_depth(self):
        from matplotlib import pyplot as plt

        x_loc_to_interpolate = [4.521, 4.521, 4.521]
        y_loc_to_interpolate = [1.196, 1.696, 2.196]

        # Vacondio simulation results for this case
        fname_vacondio_results = ['tsu_sensor1_vacondio.csv',
                                  'tsu_sensor2_vacondio.csv',
                                  'tsu_sensor3_vacondio.csv']

        output_compare_files_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'files_for_output_comparison'
                )
        fname_exp_results = os.path.join(output_compare_files_dir,
                                         'tsu_experimental.csv')
        # Experimental values of relative depth at 3 sensor locations
        # corresponding to the coordinates (4.521 m, 1.196 m), (4.521 m, 1.696
        # m), and (4.521 m, 2.196 m)
        # Note: The unit of relative depth in the file is in cm
        t_exp, dw_rel_sens1_exp, dw_rel_sens2_exp, dw_rel_sens3_exp = loadtxt(
                fname_exp_results, delimiter=',', unpack=True)
        dw_relative_sensors_exp = [dw_rel_sens1_exp, dw_rel_sens2_exp,
                                   dw_rel_sens3_exp]

        kernel = CubicSpline(dim=2)
        for n, (x_loc, y_loc) in enumerate(zip(x_loc_to_interpolate,
                                               y_loc_to_interpolate)):
            t_arr = []
            dw_arr = []
            for fname in self.output_files:
                data = load(fname)
                fluid = data['arrays']['fluid']
                t_arr.append(data['solver_data']['t'])
                interp = Interpolator([fluid], kernel=kernel)
                interp.set_interpolation_points(x_loc,
                                                y_loc)
                dw_interp = interp.interpolate('dw')
                dw_arr.append(dw_interp)

            dw_arr = array(dw_arr)
            t_arr = array(t_arr)

            plt.clf()
            if (
                self.dw0 == 0.13535 and
                self.le == 5.448 and
                self.w == 3.402 and
                self.n == 0.025 and
                self.Vb == 1.96e-4
            ):
                fname_vacondio = os.path.join(output_compare_files_dir,
                                              fname_vacondio_results[n])
                t_vac, dw_relative_vac = loadtxt(fname_vacondio,
                                                 delimiter=',',
                                                 unpack=True)

                # Converting the unit of relative depth from cm to m
                dw_relative_exp = dw_relative_sensors_exp[n] / 100.0

                plt.plot(t_vac, dw_relative_vac, 'g.', label='Vacondio et. al')
                plt.plot(t_exp, dw_relative_exp, 'k-',
                         label='Experimental data')
                fname_res = os.path.join(
                    self.output_dir, 'results_sensor%d.npz' % (n+1))
                savez(
                    fname_res,
                    x_sensor=x_loc_to_interpolate[n],
                    y_sensor=y_loc_to_interpolate[n],
                    t_numerical=t_arr, dw_relative_numerical=dw_arr-dw_arr[0],
                    t_vacondio=t_vac, dw_relative_vacondio=dw_relative_vac,
                    t_experimental=t_exp,
                    dw_relative_experimental=dw_relative_exp
                    )

            plt.plot(t_arr, dw_arr-dw_arr[0], 'r.', label='SWESPH')
            plt.legend()
            plt.ylim(-0.015, 0.05)
            plt.xlim(0, max(t_arr))
            plt.xlabel('t (s)')
            plt.ylabel('$dw-dw_0$')
            plt.title(
                    'Relative water depth at the location (%.3f m, %.3f m)' % (
                     x_loc, y_loc)
                    )
            fig = os.path.join(
                    self.output_dir,
                    "relative_depth_sensor"+str(n+1))
            plt.savefig(fig, dpi=300)


def compute_fluid_elevation(particles):
    one_time_equations = [
       Group(
            equations=[
                FluidBottomElevation(dest='fluid', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                GradientCorrectionPreStep(dest='bed', sources=['bed']),
                ]
            ),
       Group(
            equations=[
                GradientCorrection(dest='bed', sources=['bed']),
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


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        SWEOS(dest='fluid'),
                        ]
                    ),
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == "__main__":
    app = OkushiriTsunami()
    app.run()
    app.post_process(app.info_filename)
