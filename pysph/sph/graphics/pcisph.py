"""
Predictive-Corrective Incompressible SPH (PCISPH)
#################################################

References
-----------

    .. [SolPaj2009] B. Solenthaler, R. Pajarola "Predictive-Corrective
        Incompressible SPH", ACM Trans. Graph 28 (2009), pp. 1--6.

"""

import numpy as np
from compyle.api import declare
from pysph.sph.integrator import Integrator
from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator


def get_particle_array_pcisph(constants=None, **props):
    pcisph_props = [
        'p', 'rho_err', 'V',
        'au_np', 'av_np', 'aw_np', 'au_p', 'av_p', 'aw_p',
        'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0',
    ]
    pa = get_particle_array(
        constants=constants, additional_props=pcisph_props, **props
    )
    pa.add_constant('iters', 0)
    pa.add_property('num_nbrs')
    pa.add_output_arrays([
        'p', 'V', 'au_np', 'av_np', 'aw_np', 'au_p', 'av_p', 'aw_p',
        'num_nbrs', 'rho_err',
    ])
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
    def initialize(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_x0, d_y0,
                   d_z0, d_u0, d_v0, d_w0, d_rho, d_rho0, dt):
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_au_p, d_av_p, d_aw_p, d_x, d_y, d_z, d_u,
               d_v, d_w, dt, d_au_np, d_av_np, d_aw_np, d_x0, d_y0,
               d_z0, d_u0, d_v0, d_w0,):
        d_u[d_idx] = d_u0[d_idx] + dt * (d_au_p[d_idx] + d_au_np[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dt * (d_av_p[d_idx] + d_av_np[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dt * (d_aw_p[d_idx] + d_aw_np[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]


class MomentumEquationViscosity(Equation):
    def __init__(self, dest, sources, dim, nu=0.0):
        self.nu = nu
        self.dim = dim
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au_np, d_av_np, d_aw_np):
        d_au_np[d_idx] = 0.0
        d_av_np[d_idx] = 0.0
        d_aw_np[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_au_np, d_av_np,
            d_aw_np, DWIJ, XIJ, VIJ, R2IJ, EPS):
        mb = s_m[s_idx]
        rhoij = (d_rho[d_idx] + s_rho[s_idx])

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        tmp = mb * (2 + self.dim) * self.nu * xdotdwij / (rhoij * (R2IJ + EPS))

        d_au_np[d_idx] += tmp * VIJ[0]
        d_av_np[d_idx] += tmp * VIJ[1]
        d_aw_np[d_idx] += tmp * VIJ[2]


class MomentumEquationArtificialViscosity(Equation):
    def __init__(self, dest, sources, c0, alpha=0.1):
        self.alpha = alpha
        self.c0 = c0
        super(MomentumEquationArtificialViscosity, self).__init__(
            dest, sources
        )

    def initialize(self, d_idx, d_au_np, d_av_np, d_aw_np):
        d_au_np[d_idx] = 0.0
        d_av_np[d_idx] = 0.0
        d_aw_np[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_au_np, d_av_np, d_aw_np,
             RHOIJ1, R2IJ, EPS, DWIJ, VIJ, XIJ, HIJ):
        vdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vdotrij < 0:
            muij = (HIJ * vdotrij)/(R2IJ + EPS)

            piij = -self.alpha*self.c0*muij
            piij = s_m[s_idx] * piij*RHOIJ1

        d_au_np[d_idx] += -piij * DWIJ[0]
        d_av_np[d_idx] += -piij * DWIJ[1]
        d_aw_np[d_idx] += -piij * DWIJ[2]


class MomentumEquationBodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationBodyForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_au_np, d_av_np, d_aw_np):
        d_au_np[d_idx] += self.gx
        d_av_np[d_idx] += self.gy
        d_aw_np[d_idx] += self.gz


class BoundaryVolume(Equation):
    def initialize(self, d_idx, d_V, d_num_nbrs):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_num_nbrs, WIJ):
        d_V[d_idx] += WIJ

    def post_loop(self, d_idx, d_num_nbrs, d_V):
        if d_V[d_idx] > 1e-9:
            d_V[d_idx] = 1.0/d_V[d_idx]
        else:
            d_V[d_idx] = 0.0


class ComputeDensity(Equation):
    def initialize(self, d_idx, d_rho, d_num_nbrs):
        d_rho[d_idx] = 0.0
        d_num_nbrs[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_m, d_num_nbrs, WIJ):
        d_rho[d_idx] += d_m[d_idx] * WIJ
        d_num_nbrs[d_idx] += 1.0


class ComputeDensitySolids(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeDensitySolids, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_V, d_num_nbrs, WIJ):
        d_rho[d_idx] += self.rho0 * s_V[s_idx] * WIJ
        d_num_nbrs[d_idx] += 1.0


class PredictPressure(Equation):
    def __init__(self, dest, sources, alpha, rho0):
        self.rho0 = rho0
        self.alpha = alpha
        super(PredictPressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho_err, d_rho, d_p, dt):
        dt2 = dt*dt
        kpcisph = self.alpha/dt2

        rho_change = max(d_rho[d_idx], self.rho0)
        d_rho_err[d_idx] = rho_change - self.rho0
        d_p[d_idx] += kpcisph * d_rho_err[d_idx]


class UpdateVelandPos(Equation):
    def initialize(self, d_idx, d_au_p, d_av_p, d_aw_p, d_x, d_y, d_z, d_u,
               d_v, d_w, dt, d_au_np, d_av_np, d_aw_np, d_x0, d_y0,
               d_z0, d_u0, d_v0, d_w0,):
        d_u[d_idx] = d_u0[d_idx] + dt * (d_au_p[d_idx] + d_au_np[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dt * (d_av_p[d_idx] + d_av_np[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dt * (d_aw_p[d_idx] + d_aw_np[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]


class InitializePressure(Equation):
    def __init__(self, dest, sources, alpha, rho0, dim):
        self.rho0 = rho0
        self.dim = dim
        self.alpha = alpha
        super(InitializePressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_au_p, d_av_p, d_aw_p):
        d_p[d_idx] = 0.0

        d_au_p[d_idx] = 0.0
        d_av_p[d_idx] = 0.0
        d_aw_p[d_idx] = 0.0


class PressureConvergence(Equation):
    def __init__(self, dest, sources, rho0, tolerance, max_iterations, debug):
        self.conv = 0.0
        self.rho0 = rho0
        self.tolerance = tolerance
        self.debug = debug
        self.count = 0
        self.max_iterations = max_iterations
        super(PressureConvergence, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        import numpy as np

        self.count += 1
        dst.iters[0] = self.count

        rho_err_avg = dst.rho_err.mean()
        eta = self.tolerance * 0.01 * self.rho0
        if self.debug:
            print("rho_err_avg = ", rho_err_avg, "eta = ", eta)
        if abs(rho_err_avg) <= eta:
            self.conv = 1
        else:
            self.conv = -1

    def converged(self):
        if self.conv == 1 and self.count < self.max_iterations:
            self.count = 0
        if self.count >= self.max_iterations:
            self.count = 0
            print(" Pressure solve: Max iterations exceeded")
        return self.conv


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au_p, d_av_p, d_aw_p):
        d_au_p[d_idx] = 0.0
        d_av_p[d_idx] = 0.0
        d_aw_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p,
             d_au_p, d_av_p, d_aw_p, DWIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pi = d_p[d_idx]
        pj = s_p[s_idx]

        tmp = s_m[s_idx] * (pi/rhoi2 + pj/rhoj2)

        d_au_p[d_idx] -= tmp * DWIJ[0]
        d_av_p[d_idx] -= tmp * DWIJ[1]
        d_aw_p[d_idx] -= tmp * DWIJ[2]


class SolidWallBoundaryConditionAkinci(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(SolidWallBoundaryConditionAkinci, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, d_rho, d_p, d_au_p, d_av_p,
             d_aw_p, s_V, DWIJ, EPS):
        pi = d_p[d_idx]
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]

        tmp = self.rho0 * s_V[s_idx] * pi/rhoi2

        d_au_p[d_idx] -= tmp * DWIJ[0]
        d_av_p[d_idx] -= tmp * DWIJ[1]
        d_aw_p[d_idx] -= tmp * DWIJ[2]


class ComputeFactor(Equation):
    def initialize(self, d_idx, d_fac1, d_fac2, d_alpha):
        d_fac1[0] = 0.0
        d_fac1[1] = 0.0
        d_fac1[2] = 0.0

        d_fac2[0] = 0.0
        d_alpha[0] = 0.0

    def loop(self, d_idx, d_fac1, d_fac2, DWIJ):
        d_fac1[0] += DWIJ[0]
        d_fac1[1] += DWIJ[1]
        d_fac1[2] += DWIJ[2]

        dwdotdw = DWIJ[0]*DWIJ[0] + DWIJ[1]*DWIJ[1] + DWIJ[2]*DWIJ[2]
        d_fac2[0] += dwdotdw

    def post_loop(self, d_idx, d_fac1, d_fac2, d_V0, d_alpha):
        tmp  = d_fac1[0]**2 + d_fac1[1]**2 + d_fac1[2]**2
        tmp += d_fac2[0]

        beta = 2.0*d_V0[d_idx]*d_V0[d_idx]
        d_alpha[0] = 1.0 / (beta*tmp + 1e-9)


def _compute_factor(dx, h, dim, dt, kernel):
    vol = dx**dim
    _x = np.arange(-10*dx, 10*dx, dx)
    x0, y0, z0 = [0], [0], [0]

    if dim == 1:
        x = np.meshgrid(_x)
        fluid = get_particle_array(name='fluid', x=x0, V0=vol, h=h)
        dummy = get_particle_array(name='dummy', x=x, h=h)
    elif dim == 2:
        x, y = np.meshgrid(_x, _x)
        fluid = get_particle_array(name='fluid', x=x0, y=y0, V0=vol, h=h)
        dummy = get_particle_array(name='dummy', x=x, y=y, h=h)
    elif dim == 3:
        x, y, z = np.meshgrid(_x, _x, _x)
        fluid = get_particle_array(name='fluid', x=x0, y=y0, z=z0, V0=vol, h=h)
        dummy = get_particle_array(name='dummy', x=x, y=y, z=z, h=h)

    fluid.add_constant('fac1', [0.0, 0.0, 0.0])
    fluid.add_constant('fac2', [0.0])
    fluid.add_constant('alpha', [0.0])

    eq = [ComputeFactor(dest='fluid', sources=['dummy'])]

    sph_eval = SPHEvaluator(
        arrays=[fluid, dummy], equations=eq, dim=dim,
        kernel=kernel
    )
    sph_eval.evaluate()
    print("The scaling factor for PCISPH is:", fluid.alpha[0]/dt/dt)
    return fluid.alpha[0]


class PCISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, dx, h, nu, c0,
                 gx=0.0, gy=0.0, gz=0.0,
                 tolerance=1, debug=False, min_iterations=1,
                 max_iterations=1000, alpha=0.0):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.dim = dim
        self.rho0 = rho0
        self.dx = dx
        self.h = h
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tolerance = tolerance
        self.alpha = alpha
        self.c0 = c0
        self.debug = debug
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    def add_user_options(self, group):
        group.add_argument(
            '--tol', action='store', type=float, dest='tolerance',
            default=None,
            help='PCISPH relative error tolerance for convergence as a percentage.'
        )
        group.add_argument(
            '--alpha', action='store', type=float, dest='alpha',
            default=None,
            help='Artificial viscosity parameter.'
        )

        group.add_argument(
            '--max-iters', action='store', type=int,
            dest='max_iterations', default=None,
            help='Max iterations for the PPE solver'
        )
        add_bool_argument(
            group, 'debug', dest='debug', default=None,
            help="Produce some debugging output on convergence of iterations."
        )

    def consume_user_options(self, options):
        vars = ['tolerance', 'debug', 'max_iterations']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline, CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = PCISPHStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        cls = PCISPHIntegrator if integrator_cls is None else integrator_cls
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel,
            **kw
        )

    def _get_no_pressure_force(self):
        all = self.fluids + self.solids

        eq1, grp = [], []
        for fluid in self.fluids:
            if self.nu > 0.0:
                eq1.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids, dim=self.dim,
                        nu=self.nu
                    )
                )
            if ((abs(self.gx) > 0.0) or (abs(self.gy) > 0.0) or
                (abs(self.gz) > 0.0)):
                eq1.append(
                    MomentumEquationBodyForce(
                        dest=fluid, sources=None, gx=self.gx, gy=self.gy,
                        gz=self.gz
                    )
                )
            if self.alpha > 0.0:
                eq1.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=self.fluids, c0=self.c0,
                        alpha=self.alpha, 
                    )
                )
        if eq1 is not None:
            grp.append(Group(equations=eq1))
            return grp
        return []

    def _compute_density(self):
        all = self.fluids + self.solids

        grp = []
        if self.solids:
            eqns = []
            for solid in self.solids:
                # Sources should be solids or the particles leak.
                eqns.append(
                    BoundaryVolume(dest=solid, sources=self.solids)
                )
            grp.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                ComputeDensity(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDensitySolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grp.append(Group(equations=eqns))
        return grp

    def _pressure_solve(self):
        all = self.fluids + self.solids

        eqns, grps = [], []
        for fluid in self.fluids:
            eqns.append(
                InitializePressure(
                    dest=fluid, sources=None, rho0=self.rho0,
                    alpha=self.pcisph_factor, dim=self.dim
                )
            )
        grps.append(Group(equations=eqns))

        grp2 = []

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                UpdateVelandPos(dest=fluid, sources=None)
            )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                ComputeDensity(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDensitySolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PredictPressure(
                    dest=fluid, sources=None,
                    alpha=self.pcisph_factor, rho0=self.rho0
                )
            )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids
                )
            )
            if self.solids:
                eqns.append(
                    SolidWallBoundaryConditionAkinci(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PressureConvergence(
                    dest=fluid, sources=all, rho0=self.rho0,
                    tolerance=self.tolerance,
                    max_iterations=self.max_iterations, debug=self.debug
                )
            )
        grp2.append(Group(equations=eqns))

        grps.append(
            Group(equations=grp2, iterate=True,
                  min_iterations=self.min_iterations,
                  max_iterations=self.max_iterations)
        )
        return grps

    def get_equations(self):
        stages = []
        stages.extend(self._compute_density())
        stages.extend(self._get_no_pressure_force())

        self.pcisph_factor = _compute_factor(
            self.dx, self.h, self.dim, self.solver.dt, self.solver.kernel
        )
        stages.extend(self._pressure_solve())
        return stages

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

        solid_props = ['V', 'num_nbrs']
        all_solids = self.solids
        for solid in all_solids:
            pa = particle_arrays[solid]
            for prop in solid_props:
                pa.add_property(prop)
            pa.add_output_arrays(['V'])
