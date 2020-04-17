"""
Predictive-Corrective Incompressible SPH (PCISPH)
#################################################

References
-----------

    .. [SolPaj2009] B. Solenthaler, R. Pajarola "Predictive-Corrective
        Incompressible SPH", ACM Trans. Graph 28 (2009), pp. 1--6.

"""


import numpy as np
from pysph.sph.integrator import Integrator
from pysph.sph.equation import Equation, Group
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme, add_bool_argument


def get_particle_array_pcisph(constants=None, **props):
    pcisph_props = [
        'au', 'av', 'aw', 'arho', 'dwij2', 'u0', 'v0', 'w0', 'aup', 'avp',
        'awp', 'x0', 'y0', 'z0', 'rho0'
    ]
    pa = get_particle_array(
        constants=constants, additional_props=pcisph_props, **props
    )
    pa.add_constant('iters', np.zeros(10000))
    pa.add_property('dw', stride=3)
    pa.add_output_arrays(['p', 'dwij2'])
    return pa


class PCISPHIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations(0)

        self.stage1()
        self.update_domain()

        self.do_post_stage(dt, 1)

    def initial_acceleration(self, t, dt):
        pass


class PCISPHStep(IntegratorStep):
    def __init__(self, show_itercount=False):
        self.show_itercount = show_itercount
        self.index = 0

    def initialize(self, d_idx, d_u, d_v, d_w, d_u0, d_v0, d_w0,
                   d_x, d_y, d_z, d_x0, d_y0, d_z0, d_rho, d_rho0):
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def py_stage1(self, dst, t, dt):
        if self.show_itercount:
            print("Iteration count = ", dst.iters[self.index])
            self.index += 1

    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_x, d_y, d_z,
               d_aup, d_avp, d_awp, d_u0, d_v0, d_w0, d_x0, d_y0, d_z0, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * (d_au[d_idx] + d_aup[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dt * (d_av[d_idx] + d_avp[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dt * (d_aw[d_idx] + d_awp[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]


class MomentumEquationViscosity(Equation):
    r"""**Momentum Equation Viscosity**

    See "pysph.sph.wc.viscosity.LaminarViscocity"
    """

    def __init__(self, dest, sources, nu=0.0, gx=0.0, gy=0.0, gz=0.0):
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_au, d_av, d_aw, DWIJ,
             XIJ, VIJ, R2IJ, EPS):
        mb = s_m[s_idx]
        rhoij = (d_rho[d_idx] + s_rho[s_idx])

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        tmp = mb * 4 * self.nu * xdotdwij / (rhoij * (R2IJ + EPS))

        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_u, d_v, d_w, d_p, d_aup,
                  d_avp, d_awp, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        # Retaining the old pressure seems to give better results for the
        # TG problem.
        #d_p[d_idx] = 0.0

        d_aup[d_idx] = 0.0
        d_avp[d_idx] = 0.0
        d_awp[d_idx] = 0.0


class Predict(Equation):
    r"""**Predict velocity and position**

    .. math::
        \mathbf{v}^{*}(t+1) = \mathbf{v}(t) + dt
        \left(\frac{d \mathbf{v}_{visc, g}(t)}{dt} +
        \frac{d \mathbf{v}_{p} (t)}{dt} \right)

    .. math::
        \mathbf{x}^{*}(t+1) = \mathbf{x}(t) + dt * \mathbf{v}(t+1)
    """

    def initialize(self, d_idx, d_u, d_v, d_w, d_aup, d_avp, d_awp, d_x, d_y,
                   d_z, d_au, d_av, d_aw, d_u0, d_v0, d_w0, d_x0, d_y0, d_z0,
                   dt):
        d_u[d_idx] = d_u0[d_idx] + dt * (d_au[d_idx] + d_aup[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dt * (d_av[d_idx] + d_avp[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dt * (d_aw[d_idx] + d_awp[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]


class ComputePressure(Equation):
    r"""**Compute Pressure**

    Compute pressure iteratively maintaining density within a given tolerance.

    .. math::
        p_i += \delta \rho^{*}_{{err}_i}

    where,

    .. math::
        \rho_{err_i} = \rho_i^{*} - \rho_0

    .. math::
        \delta = \frac{-1}{\beta (-\sum_j \nabla W_{ij} \cdot
        \sum_j \nabla W_{ij} - \sum_j \nabla W_{ij} \nabla W_{ij})}
    """

    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputePressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dw, d_dwij2):
        d_dw[d_idx * 3 + 0] = 0.0
        d_dw[d_idx * 3 + 1] = 0.0
        d_dw[d_idx * 3 + 2] = 0.0

        d_dwij2[d_idx] = 0.0

    def loop(self, d_idx, d_dw, d_dwij2, DWIJ):
        d_dw[d_idx * 3 + 0] += DWIJ[0]
        d_dw[d_idx * 3 + 1] += DWIJ[1]
        d_dw[d_idx * 3 + 2] += DWIJ[2]

        dwij2 = DWIJ[0] * DWIJ[0] + DWIJ[1] * DWIJ[1] + DWIJ[2] * DWIJ[2]
        d_dwij2[d_idx] += dwij2

    def post_loop(self, d_idx, d_dw, d_m, dt, d_dwij2, d_p, d_rho):
        dwx = d_dw[d_idx * 3 + 0]
        dwy = d_dw[d_idx * 3 + 1]
        dwz = d_dw[d_idx * 3 + 2]
        tmp = dwx * dwx + dwy * dwy + dwz * dwz

        mi = d_m[d_idx]
        rho0 = self.rho0
        beta = 2 * mi * mi * (dt / rho0) * (dt / rho0)
        delta = 1.0 / (beta * (tmp + d_dwij2[d_idx]))

        rho_err = d_rho[d_idx] - rho0
        d_p[d_idx] += delta * rho_err


class MomentumEquationPressureGradient(Equation):
    r"""**Momentum Equation pressure gradient**

    Standard WCSPH pressure gradient,

    .. math::
        \frac{d\mathbf{v}}{dt} = - \sum_j m_j \left(\frac{p_i}{\rho_i^2}
        + \frac{p_i}{\rho_i^2}\right) \nabla W(x_{ij}, h)
    """

    def __init__(self, dest, sources, rho0, tolerance, debug):
        self.rho0 = rho0
        self.tolerance = tolerance
        self.debug = debug
        self.rho_err = 0.0
        self.ctr = 0
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_p, s_p, d_rho, s_rho, s_m, d_aup, d_avp,
             d_awp, DWIJ):
        rhoi2 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj2 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])
        mj = s_m[d_idx]

        pij = -1.0 * mj * (d_p[d_idx] * rhoi2 + s_p[s_idx] * rhoj2)
        d_aup[d_idx] += pij * DWIJ[0]
        d_avp[d_idx] += pij * DWIJ[1]
        d_awp[d_idx] += pij * DWIJ[2]

    def reduce(self, dst, t, dt):
        import numpy as np
        self.rho_err = np.mean(np.abs(dst.rho / self.rho0 - 1.0))
        dst.iters[self.ctr] += 1

    def converged(self):
        debug = self.debug
        rho_err = self.rho_err

        if rho_err > self.tolerance:
            if debug:
                print("Not converged:", rho_err)
            return -1.0
        else:
            self.ctr += 1
            if debug:
                print("Converged:", rho_err)
            return 1.0


class PCISPHScheme(Scheme):
    def __init__(self, fluids, dim, rho0, nu, gx=0.0, gy=0.0, gz=0.0,
                 tolerance=0.1, debug=False, show_itercount=False):
        self.fluids = fluids
        self.solver = None
        self.dim = dim
        self.rho0 = rho0
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tolerance = tolerance
        self.debug = debug
        self.show_itercount = show_itercount

    def add_user_options(self, group):
        group.add_argument(
            '--pcisph-tol', action='store', type=float, dest='tolerance',
            default=None,
            help='relative error tolerance for convergence as a percentage.'
        )
        add_bool_argument(
            group, 'pcisph-debug', dest='debug', default=None,
            help="Produce some debugging output on convergence of iterations."
        )
        add_bool_argument(
            group, 'pcisph-itercount', dest='show_itercount', default=False,
            help="Produce some debugging output on convergence of iterations."
        )

    def consume_user_options(self, options):
        vars = ['tolerance', 'debug', 'show_itercount']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = PCISPHStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls(self.show_itercount)

        cls = PCISPHIntegrator if integrator_cls is None else integrator_cls
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel,
            **kw
        )

    def get_equations(self):
        from pysph.sph.basic_equations import SummationDensity
        all = self.fluids
        equations = []

        eq1 = []
        for fluid in self.fluids:
            eq1.append(
                MomentumEquationViscosity(
                    dest=fluid, sources=all, nu=self.nu, gx=self.gx,
                    gy=self.gy, gz=self.gz
                )
            )
        equations.append(Group(equations=eq1))

        eq1, g2 = [], []
        for fluid in self.fluids:
            eq1.append(Predict(dest=fluid, sources=None))
        g2.append(Group(equations=eq1, update_nnps=True))

        eq2 = []
        for fluid in self.fluids:
            eq2.append(SummationDensity(dest=fluid, sources=all))
        g2.append(Group(equations=eq2))

        eq3 = []
        for fluid in self.fluids:
            eq3.append(
                ComputePressure(dest=fluid, sources=all, rho0=self.rho0)
            )
        g2.append(Group(equations=eq3, update_nnps=True))

        eq4 = []
        for fluid in self.fluids:
            eq4.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, rho0=self.rho0,
                    tolerance=self.tolerance, debug=self.debug
                ),
            )
        g2.append(Group(equations=eq4))

        equations.append(
            Group(equations=g2, iterate=True,
                  max_iterations=500, min_iterations=2)
        )
        return equations

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_pcisph(name='junk')
        props = list(dummy.properties.keys())
        props += [dict(name=x, stride=y) for x, y in dummy.stride.items()]
        output_props = dummy.output_property_arrays
        constants = [dict(name=x, data=y) for x, y in dummy.constants.items()]
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            for const in constants:
                pa.add_constant(**const)
