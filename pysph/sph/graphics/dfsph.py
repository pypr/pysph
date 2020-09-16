"""
Divergence-Free SPH
#################################################

References
-----------

    .. [BenKos2017] J. Bender and D. Koschier, "Divergence-Free SPH for
        Incompressible and Viscous Fluids," in IEEE Transactions on
        Visualization and Computer Graphics, vol. 23, no. 3, pp. 1193-1206,
        1 March 2017, doi: 10.1109/TVCG.2016.2578335.

"""

import numpy as np
from compyle.api import declare
from pysph.sph.integrator import Integrator
from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator


def get_particle_array_dfsph(constants=None, **props):
    dfsph_props = [
        'au_np', 'av_np', 'aw_np', 'au_p', 'av_p', 'aw_p',
        'p', 'rho_err', 'V', 'div', 'alpha',
    ]
    pa = get_particle_array(
        constants=constants, additional_props=dfsph_props, **props
    )
    pa.add_constant('rho_iters', 0)
    pa.add_constant('div_iters', 0)
    pa.add_property('num_nbrs')
    pa.add_property('fac1', stride=3)
    pa.add_property('fac2')
    pa.add_output_arrays([
        'p', 'div', 'alpha', 'V', 'au_np', 'av_np', 'aw_np', 'au_p',
        'av_p', 'aw_p',
    ])
    return pa


class DFSPHIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations(0)

        self.stage1()

        self.do_post_stage(dt, 1)

        self.compute_accelerations(1)

        self.stage2()

        self.update_domain()

        self.do_post_stage(dt, 1)

    def initial_acceleration(self, t, dt):
        pass


class DFSPHStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au_np, d_av_np, d_aw_np, dt):
        d_u[d_idx] = d_u[d_idx] + dt * d_au_np[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_av_np[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_aw_np[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]


class MomentumEquationViscosity(Equation):
    def __init__(self, dest, sources, dim, nu):
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
    def __init__(self, dest, sources, c0, alpha):
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
    def __init__(self, dest, sources, gx, gy, gz):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationBodyForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_au_np, d_av_np, d_aw_np):
        d_au_np[d_idx] += self.gx
        d_av_np[d_idx] += self.gy
        d_aw_np[d_idx] += self.gz


class BoundaryVolume(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

    def post_loop(self, d_idx, d_V):
        if d_V[d_idx] > 1e-9:
            d_V[d_idx] = 1.0/d_V[d_idx]
        else:
            d_V[d_idx] = 0.0


class ComputeDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_m, WIJ):
        d_rho[d_idx] += d_m[d_idx] * WIJ


class ComputeDensitySolids(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeDensitySolids, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_V, WIJ):
        d_rho[d_idx] += self.rho0 * s_V[s_idx] * WIJ


class ComputeDivergence(Equation):
    def initialize(self, d_idx, d_div, d_num_nbrs):
        d_div[d_idx] = 0.0
        d_num_nbrs[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_div, d_num_nbrs, DWIJ, VIJ):
        udotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_div[d_idx] += s_m[s_idx] * udotdwij
        d_num_nbrs[d_idx] += 1.0


class ComputeDivergenceSolids(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeDivergenceSolids, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_V, d_div, d_num_nbrs, DWIJ, VIJ):
        udotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_div[d_idx] += self.rho0 * s_V[s_idx] * udotdwij
        d_num_nbrs[d_idx] += 1.0


class PredictPressure(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PredictPressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho_err, d_div, d_rho, d_p,
                   d_alpha, dt):
        dt2 = dt*dt
        kpcisph = d_alpha[d_idx]/dt2

        d_rho_err[d_idx] = d_rho[d_idx] + dt*d_div[d_idx]
        d_rho_err[d_idx] = max(d_rho_err[d_idx], self.rho0)

        d_rho_err[d_idx] = d_rho_err[d_idx] - self.rho0

        d_p[d_idx] = kpcisph * d_rho_err[d_idx]


class PredictPressureDiv(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(PredictPressureDiv, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho_err, d_div, d_p, d_num_nbrs,
                   d_alpha, dt):
        kpcisph = d_alpha[d_idx]/dt

        d_div[d_idx] = max(d_div[d_idx], 0.0)

        if self.dim > 2:
            if d_num_nbrs[d_idx] < 20:
                d_div[d_idx] = 0.0
        else:
            if d_num_nbrs[d_idx] < 7:
                d_div[d_idx] = 0.0

        d_rho_err[d_idx] = dt * d_div[d_idx]
        d_p[d_idx] = kpcisph * d_div[d_idx]


class UpdateVelocity(Equation):
    def initialize(self, d_idx, d_au_p, d_av_p, d_aw_p, d_u, d_v, d_w,
                   dt):
        d_u[d_idx] += dt * d_au_p[d_idx]
        d_v[d_idx] += dt * d_av_p[d_idx]
        d_w[d_idx] += dt * d_aw_p[d_idx]


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
        dst.rho_iters[0] = self.count

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
            print(" Rho solve: Max iterations exceeded")
        return self.conv


class PressureConvergenceDiv(Equation):
    def __init__(self, dest, sources, rho0, tolerance, max_iterations, debug):
        self.conv = 0.0
        self.rho0 = rho0
        self.tolerance = tolerance
        self.debug = debug
        self.count = 0
        self.max_iterations = max_iterations
        super(PressureConvergenceDiv, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        import numpy as np

        self.count += 1
        dst.div_iters[0] = self.count

        rho_err_avg = dst.rho_err.mean()
        eta = self.tolerance * 0.01 * self.rho0
        if self.debug:
            print("div_err_avg = ", rho_err_avg, "eta = ", eta)
        if abs(rho_err_avg) <= eta:
            self.conv = 1
        else:
            self.conv = -1

    def converged(self):
        if self.conv == 1 and self.count < self.max_iterations:
            self.count = 0
        if self.count >= self.max_iterations:
            self.count = 0
            print(" Div solve: Max iterations exceeded")
        return self.conv


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au_p, d_av_p, d_aw_p):
        d_au_p[d_idx] = 0.0
        d_av_p[d_idx] = 0.0
        d_aw_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p,
             d_au_p, d_av_p, d_aw_p, DWIJ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pi = d_p[d_idx]
        pj = s_p[s_idx]

        tmp = s_m[s_idx] * (pi/rhoi + pj/rhoj)

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
        rhoi = d_rho[d_idx]
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]

        tmp = self.rho0 * s_V[s_idx] * pi/rhoi

        d_au_p[d_idx] -= tmp * DWIJ[0]
        d_av_p[d_idx] -= tmp * DWIJ[1]
        d_aw_p[d_idx] -= tmp * DWIJ[2]


class ComputeFactors(Equation):
    def initialize(self, d_idx, d_fac1, d_fac2):
        d_fac1[d_idx*3 + 0] = 0.0
        d_fac1[d_idx*3 + 1] = 0.0
        d_fac1[d_idx*3 + 2] = 0.0
        d_fac2[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_fac1, d_fac2, DWIJ):
        d_fac1[d_idx*3 + 0] += s_m[s_idx] * DWIJ[0]
        d_fac1[d_idx*3 + 1] += s_m[s_idx] * DWIJ[1]
        d_fac1[d_idx*3 + 2] += s_m[s_idx] * DWIJ[2]

        dwdotdw = DWIJ[0]*DWIJ[0] + DWIJ[1]*DWIJ[1] + DWIJ[2]*DWIJ[2]
        d_fac2[d_idx] += s_m[s_idx] * s_m[s_idx] * dwdotdw

    def post_loop(self, d_idx, d_rho, d_alpha, d_fac1, d_fac2):
        tmp  = d_fac1[d_idx*3 + 0] * d_fac1[d_idx*3 + 0]
        tmp += d_fac1[d_idx*3 + 1] * d_fac1[d_idx*3 + 1]
        tmp += d_fac1[d_idx*3 + 2] * d_fac1[d_idx*3 + 2]

        tmp += d_fac2[d_idx]

        if tmp > 1e-6:
            d_alpha[d_idx] = d_rho[d_idx] / tmp
        else:
            d_alpha[d_idx] = 0.0


class ComputeFactorsSolids(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeFactorsSolids, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, s_V, d_fac1, DWIJ):
        tmp = self.rho0 * s_V[s_idx]
        d_fac1[d_idx*3 + 0] += tmp * DWIJ[0]
        d_fac1[d_idx*3 + 1] += tmp * DWIJ[1]
        d_fac1[d_idx*3 + 2] += tmp * DWIJ[2]


class DFSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, nu, c0, gx=0.0,
                 gy=0.0, gz=0.0, tolerance=1, debug=False, min_iterations=1,
                 max_iterations=1000, alpha=0.0):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.dim = dim
        self.rho0 = rho0
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tolerance = tolerance
        self.debug = debug
        self.alpha = alpha
        self.c0 = c0
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    def add_user_options(self, group):
        group.add_argument(
            '--tol', action='store', type=float, dest='tolerance',
            default=None,
            help='relative error tolerance for convergence as a percentage.'
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

        step_cls = DFSPHStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        cls = DFSPHIntegrator if integrator_cls is None else integrator_cls
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
                        dest=fluid, sources=self.fluids,
                        alpha=self.alpha, c0=self.c0
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

    def _compute_factors(self):
        all = self.fluids + self.solids

        eqns, grp = [], []
        for fluid in self.fluids:
            eqns.append(
                ComputeFactors(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeFactorsSolids(dest=fluid,
                        sources=self.solids, rho0=self.rho0)
                )
        grp.append(Group(equations=eqns))
        return grp

    def _divergence_solve(self):
        all = self.fluids + self.solids

        eqns, grps = [], []
        for fluid in self.fluids:
            eqns.append(
                ComputeDivergence(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDivergenceSolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grps = [Group(equations=eqns)]

        grp2 = []

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PredictPressureDiv(
                    dest=fluid, sources=None, dim=self.dim
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
                UpdateVelocity(dest=fluid, sources=None)
            )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                ComputeDivergence(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDivergenceSolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PredictPressureDiv(dest=fluid, sources=None, dim=self.dim)
            )
            eqns.append(
                PressureConvergenceDiv(
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

    def _pressure_solve(self):
        all = self.fluids + self.solids

        eqns, grps = [], []
        for fluid in self.fluids:
            eqns.append(
                ComputeDivergence(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDivergenceSolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grps = [Group(equations=eqns)]

        grp2 = []

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PredictPressure(
                    dest=fluid, sources=None, rho0=self.rho0
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
                UpdateVelocity(dest=fluid, sources=None)
            )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                ComputeDivergence(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eqns.append(
                    ComputeDivergenceSolids(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        grp2.append(Group(equations=eqns))

        eqns = []
        for fluid in self.fluids:
            eqns.append(
                PredictPressure(dest=fluid, sources=None, rho0=self.rho0)
            )
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
        stg0 = []
        stg0.extend(self._compute_density())
        stg0.extend(self._compute_factors())
        stg0.extend(self._divergence_solve())
        stg0.extend(self._get_no_pressure_force())

        stg1 = []
        stg1.extend(self._pressure_solve())
        stages = MultiStageEquations([stg0, stg1])
        return stages

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_dfsph(name='junk')
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

        solid_props = ['V']
        all_solids = self.solids
        for solid in all_solids:
            pa = particle_arrays[solid]
            for prop in solid_props:
                pa.add_property(prop)
            pa.add_output_arrays(['V'])
