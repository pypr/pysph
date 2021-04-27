"""Rectangular open channel flow over an inclined bed with friction. (4 min)

The case is described in "Shallow Water and Navier-Stokes SPH-like numerical
modelling of rapidly varying free-surface flows", Renato Vacondio, Doctoral
dissertation, University of Parma, pp 99-103 (2010).

"""
import os

# Numpy
from numpy import (ones_like, zeros_like, sort, arange, sqrt, savez)
import numpy as np

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
        ParticleAcceleration, BoundaryInnerReimannStateEval, SubCriticalInFlow,
        SubCriticalOutFlow, BedFrictionSourceEval
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator

# PySPH Inlet_Outlet
from pysph.sph.bc.donothing.simple_inlet_outlet import (
    SimpleInletOutlet)
from pysph.sph.bc.inlet_outlet_manager import (
    InletInfo, OutletInfo, OutletStep, InletStep)

# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class RectangularOpenChannelFlow(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=10.0,
            help="Spacing between the particles")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=5.0,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--le", action="store", type=float, dest="le", default=800.0,
            help="Length of the channel (m)")
        group.add_argument(
            "--w", action="store", type=float, dest="w", default=400.0,
            help="Width of the channel (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=0.0316,
            help="Manning coefficient ")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dx = self.options.dx
        self.dw0 = self.options.dw0
        self.le = self.options.le
        self.w = self.options.w
        self.n = self.options.n

        # Inlet and Outlet
        self.num_inlet_pa = 2
        self.num_outlet_pa = 3
        self.x_max_inlet = 0.
        self.x_min_inlet = -self.dx * self.num_inlet_pa
        self.x_min_outlet = self.le
        self.x_max_outlet = self.le + self.num_outlet_pa*self.dx

        # Initial Condition
        q = 14.645  # Specific Discharge
        self.u_inlet = q / self.dw0

    def create_particles(self):
        hdx = self.hdx
        dx = self.dx
        d = self.dw0
        w = self.w
        le = self.le
        # Inlet Properties
        x, y = np.mgrid[-self.num_inlet_pa*dx + dx/2.:0:dx, dx/2:w-(dx/4.):dx]
        x, y = (np.ravel(t) for t in (x, y))

        u_inlet = self.u_inlet

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d
        alpha = dim * rho
        cs = sqrt(9.8 * rho/rho_w)

        # Note: Both u and uh must be specified when specifying inlet vel
        u = ones_like(x) * u_inlet
        uh = ones_like(x) * u_inlet

        # Need to specify this to inlet open boundary particle (OBP) because
        # when OBP becomes fluid particle, the fluid particle will have this
        # bottom slope (bx)
        bx = ones_like(x) * -0.001

        inlet = gpa_swe(x=x, y=y, m=m, rho0=rho0, rho=rho, h0=h0, h=h, u=u,
                        uh=uh, alpha=alpha, cs=cs, bx=bx, name='inlet')
        boundary_props = ['dw_inner_reimann', 'u_inner_reimann',
                          'v_inner_reimann', 'shep_corr']
        inlet.add_output_arrays(boundary_props)
        inlet.add_property('x0')

        # Fluid Properties
        xf, yf = np.mgrid[0.5*dx:self.x_max_inlet+le:dx, dx/2:w-(dx/4.):dx]
        xf, yf = (np.ravel(t) for t in (xf, yf))
        m = ones_like(xf) * dx * dx * rho_w * d
        h = ones_like(xf) * hdx * dx
        h0 = ones_like(xf) * hdx * dx
        rho = ones_like(xf) * rho_w * d
        rho0 = ones_like(xf) * rho_w * d
        uh = ones_like(xf) * u_inlet
        u = ones_like(xf) * u_inlet
        bx = ones_like(xf) * -0.001
        fluid = gpa_swe(name='fluid', x=xf, y=yf, m=m, rho0=rho0, rho=rho, h=h,
                        bx=bx, h0=h0, uh=uh, u=u)

        # Outlet Properties
        xo, yo = np.mgrid[dx/2.:self.num_outlet_pa*dx:dx, dx/2:w-(dx/4.):dx]
        xo, yo = (np.ravel(t) for t in (xo, yo))
        xo += le
        dw = ones_like(xo) * d
        m = ones_like(xo) * dx * dx * rho_w * d
        h = ones_like(xo) * hdx * dx
        h0 = ones_like(xo) * hdx * dx
        rho = ones_like(xo) * rho_w * d
        rho0 = ones_like(xo) * rho_w * d
        cs = sqrt(9.8 * rho/rho_w)
        alpha = dim * rho
        outlet = gpa_swe(name='outlet', x=xo, y=yo, dw=dw, m=m, rho0=rho0,
                         alpha=alpha, rho=rho, h=h, h0=h0, cs=cs)
        outlet.add_output_arrays(boundary_props)
        outlet.add_property('x0')

        # Bed Properties
        xb, yb = np.mgrid[-5*dx:le*1.6+5*dx:dx, 0:w+dx/2.:dx]
        xb = np.ravel(xb)
        yb = np.ravel(yb)

        Vb = ones_like(xb) * dx * dx
        nb = ones_like(xb) * self.n  # Manning Coefficient
        hb = ones_like(xb) * hdx * dx

        bed = gpa_swe(name='bed', x=xb, y=yb, V=Vb, n=nb, h=hb)

        # Closed Boundary
        xcb_top = np.arange(self.x_min_inlet-2.0*dx,
                            self.x_max_outlet*1.6, dx)
        ycb_top = np.concatenate((ones_like(xcb_top)*(w+0.5*dx),
                                  ones_like(xcb_top)*(w+1.5*dx)), axis=0)
        xcb_top = np.tile(xcb_top, 2)

        xcb_bottom = np.arange(self.x_min_inlet-2.0*dx,
                               self.x_max_outlet*1.6, dx)
        ycb_bottom = np.concatenate((zeros_like(xcb_bottom)-0.5*dx,
                                     zeros_like(xcb_bottom)-1.5*dx), axis=0)
        xcb_bottom = np.tile(xcb_bottom, 2)

        xcb_all = np.concatenate((xcb_top, xcb_bottom), axis=0)
        ycb_all = np.concatenate((ycb_top, ycb_bottom), axis=0)
        m_cb = ones_like(xcb_all) * dx * dx * rho_w * d
        h_cb = ones_like(xcb_all) * hdx * dx
        rho_cb = ones_like(xcb_all) * rho_w * d
        dw_cb = ones_like(xcb_all) * d
        cs_cb = sqrt(9.8 * dw_cb)
        alpha_cb = dim * rho_cb
        u_cb = ones_like(xcb_all) * u_inlet
        is_wall_boun_pa = ones_like(xcb_all)

        boundary = gpa_swe(name='boundary', x=xcb_all, y=ycb_all, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb, u=u_cb,
                           is_wall_boun_pa=is_wall_boun_pa)
        return [inlet, fluid, outlet, bed, boundary]

    def _create_inlet_outlet_manager(self):
        from pysph.sph.bc.donothing.inlet import Inlet
        from pysph.sph.bc.donothing.outlet import Outlet

        props_to_copy = ['x', 'y', 'u', 'v', 'w', 'm',
                         'h', 'rho', 'rho0', 'bx', 'h0', 'uh']
        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[self.x_min_inlet, 0.0, 0.0], has_ghost=False,
            update_cls=Inlet
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[self.x_max_outlet, 0.0, 0.0], update_cls=Outlet,
            props_to_copy=props_to_copy
        )

        iom = SimpleInletOutlet(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )

        return iom

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        compute_initial_props(list(particle_arrays.values()))
        io = iom.get_inlet_outlet(particle_arrays)
        return io

    def create_solver(self):
        self.iom = self._create_inlet_outlet_manager()
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(inlet=InletStep(), fluid=SWEStep(),
                                   outlet=OutletStep())
        tf = 100
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.1,
            adaptive_timestep=True,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    GatherDensityEvalNextIteration(
                        dest='fluid', sources=['inlet', 'fluid',
                                               'outlet', 'boundary']),
                    NonDimensionalDensityResidual(dest='fluid'),
                    UpdateSmoothingLength(dim=dim, dest='fluid'),
                    CheckConvergenceDensityResidual(dest='fluid')
                ], iterate=True, max_iterations=10
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid', 'inlet', 'outlet',
                                               'boundary']),
                    SWEOS(dest='fluid')
                ]
            ),
            Group(
                equations=[
                    BoundaryInnerReimannStateEval(dest='inlet',
                                                  sources=['fluid']),
                    BoundaryInnerReimannStateEval(dest='outlet',
                                                  sources=['fluid'])
                    ]
                ),
            Group(
                equations=[
                    SubCriticalInFlow(dest='inlet'),
                    SubCriticalOutFlow(dest='outlet')
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
                    ParticleAcceleration(dim=dim, dest='fluid',
                                         sources=['fluid', 'inlet', 'outlet',
                                                  'boundary'])
                    ]
                ),
            ]
        return equations

    def post_step(self, solver):
        for pa in self.particles:
            if pa.name == 'outlet':
                o_pa = pa
        arr_ones = ones_like(o_pa.rho)
        o_pa.alpha = arr_ones * dim * rho_w * self.dw0
        o_pa.rho = arr_ones * rho_w * self.dw0
        o_pa.dw = arr_ones * self.dw0
        o_pa.cs = sqrt(9.8 * o_pa.dw)

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        import matplotlib
        matplotlib.use('Agg')
        self._plot_vel_and_depth_at_channel_mid_section()

    def _plot_vel_and_depth_at_channel_mid_section(self):
        from matplotlib import pyplot as plt
        t_to_plot = [100.]
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

            y = arange(self.dx/2., self.w-(self.dx/4.), self.dx)
            y_srt = sort(y)
            len_y = len(y_srt)
            idx_mid_y = int(len_y/2.)
            mid_y = y_srt[idx_mid_y]
            cond = abs(fluid.y - mid_y) < 1e-2

            x = fluid.x[cond]
            y = fluid.y[cond]
            dw = fluid.dw[cond]
            rho = fluid.rho[cond]
            u = fluid.u[cond]
            v = fluid.v[cond]
            au = fluid.au[cond]
            av = fluid.av[cond]

            if len(x) == 0:
                return
            x_ind_sort = x.argsort()
            x = x[x_ind_sort]
            y = y[x_ind_sort]
            x_max = max(x)
            dw = dw[x_ind_sort]
            rho = rho[x_ind_sort]
            u = u[x_ind_sort]
            v = v[x_ind_sort]
            au = au[x_ind_sort]
            av = av[x_ind_sort]

            dw_actual = ones_like(x) * self.dw0
            u_actual = ones_like(x) * self.u_inlet

            fname_res = os.path.join(
                    self.output_dir, 'results.npz')
            savez(
                fname_res, t=t_to_plot[num], x=x, y=y,
                dw=dw, u=u, v=v, rho=rho, au=au, av=av)

            # Plot properties of the fluid domain
            plt.clf()
            plt.plot(x, dw_actual, 'r-', label='Exact Solution')
            plt.plot(x, dw, 'bo', label='SWE-SPH', markersize=2)
            plt.ylim(0.5*self.dw0, 1.5*self.dw0)
            plt.xlim(0, x_max+self.dx)
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('dw (m)')
            plt.title('t = %0.1fs' % t_to_plot[num])
            fig = os.path.join(self.output_dir, "depth%d.png" % (num+1))
            plt.savefig(fig, dpi=300)

            plt.clf()
            plt.plot(x, u_actual, 'r-', label='Exact Solution')
            plt.plot(x, u, 'bo', label='SWE-SPH', markersize=2)
            plt.ylim(0.5*self.u_inlet, 1.5*self.u_inlet)
            plt.xlim(0, x_max+self.dx)
            plt.legend()
            plt.xlabel('x (m)')
            plt.ylabel('u (m/s)')
            plt.title('t = %0.1fs' % t_to_plot[num])
            fig = os.path.join(self.output_dir, "velocity%d.png" % (num+1))
            plt.savefig(fig, dpi=300)
            num += 1


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        SWEOS(dest='fluid'),
                        ]
                    ),
                Group(
                    equations=[
                        BoundaryInnerReimannStateEval(dest='inlet',
                                                      sources=['fluid']),
                        BoundaryInnerReimannStateEval(dest='outlet',
                                                      sources=['fluid'])
                        ]
                    ),
                Group(
                    equations=[
                        SubCriticalInFlow(dest='inlet'),
                        SubCriticalOutFlow(dest='outlet')
                        ]
                    ),
                Group(
                    equations=[
                        CorrectionFactorVariableSmoothingLength(
                            dest='fluid', sources=['fluid', 'inlet', 'outlet',
                                                   'boundary'])
                        ]
                    ),
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == "__main__":
    app = RectangularOpenChannelFlow()
    app.run()
    app.post_process(app.info_filename)
