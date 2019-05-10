"""
Incompressible SPH
"""
import numpy
import numpy as np
from compyle.api import declare
from pysph.sph.scheme import Scheme
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation, Group, MultiStageEquations


def get_particle_array_isph(constants=None, **props):
    isph_props = [
        'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0', 'diag', 'odiag',
        'pk', 'rhs', 'pdiff', 'wg', 'vf', 'vg', 'ug', 'wij', 'wf', 'uf',
        'V', 'au_visc', 'av_visc', 'aw_visc', 'au_pre', 'av_pre', 'aw_pre'
    ]

    # No of particles
    N = len(props['gid'])
    consts = {
        'np': np.array([N], dtype=int),
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(
        additional_props=isph_props, constants=consts, **props
    )
    pa.add_output_arrays(['p', 'V'])
    return pa


class PECIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations(0)

        self.stage1()

        self.update_domain()

        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations(1)

        self.stage2()

        self.update_domain()

        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass


class ISPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v,
                   d_w, d_u0, d_v0, d_w0, dt):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au_visc, d_av_visc,
               d_aw_visc, dt):
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_u[d_idx] += dt*d_au_visc[d_idx]
        d_v[d_idx] += dt*d_av_visc[d_idx]
        d_w[d_idx] += dt*d_aw_visc[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_u0, d_v0, d_w0,
               d_x0, d_y0, d_z0, dt, d_au_pre, d_av_pre, d_aw_pre):
        d_u[d_idx] += dt*d_au_pre[d_idx]
        d_v[d_idx] += dt*d_av_pre[d_idx]
        d_w[d_idx] += dt*d_aw_pre[d_idx]

        d_x[d_idx] = d_x0[d_idx] + 0.5*dt * (d_u[d_idx] + d_u0[d_idx])
        d_y[d_idx] = d_y0[d_idx] + 0.5*dt * (d_v[d_idx] + d_v0[d_idx])
        d_z[d_idx] = d_z0[d_idx] + 0.5*dt * (d_w[d_idx] + d_w0[d_idx])


class ISPHDIStep(ISPHStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au_visc, d_av_visc,
               d_aw_visc, dt):
        d_u[d_idx] += dt*d_au_visc[d_idx]
        d_v[d_idx] += dt*d_av_visc[d_idx]
        d_w[d_idx] += dt*d_aw_visc[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]


class ISPHDFDIStep(ISPHStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au_visc, d_av_visc,
               d_aw_visc, dt):
        d_u[d_idx] += dt*d_au_visc[d_idx]
        d_v[d_idx] += dt*d_av_visc[d_idx]
        d_w[d_idx] += dt*d_aw_visc[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]


class MomentumEquationViscosity(Equation):
    def __init__(self, dest, sources, nu, c0, alpha, beta,
                 gx=0.0, gy=0.0, gz=0.0):
        self.nu = nu
        self.c0 = c0
        self.alpha = alpha
        self.beta = beta
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au_visc, d_av_visc, d_aw_visc):
        d_au_visc[d_idx] = self.gx
        d_av_visc[d_idx] = self.gy
        d_aw_visc[d_idx] = self.gz

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_au_visc, d_av_visc,
             d_aw_visc, XIJ, DWIJ, R2IJ, EPS, VIJ, RHOIJ1, HIJ):
        nu = self.nu
        rhoij = (s_rho[s_idx] + d_rho[d_idx])
        rhoij2_1 = 1.0/(rhoij*rhoij)
        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        fac = 8.0 * s_m[s_idx] * nu * rhoij2_1 * xdotdwij / (R2IJ + EPS)

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        piij = 0.0
        if vijdotxij < 0.0:
            muij = (HIJ * vijdotxij)/(R2IJ + EPS)
            piij = -self.alpha*self.c0*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

            d_au_visc[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
            d_av_visc[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
            d_aw_visc[d_idx] += -s_m[s_idx] * piij * DWIJ[2]

        d_au_visc[d_idx] += fac * VIJ[0]
        d_av_visc[d_idx] += fac * VIJ[1]
        d_aw_visc[d_idx] += fac * VIJ[2]


class VelocityDivergence(Equation):
    def initialize(self, d_idx, d_rhs, d_pk, d_p):
        d_rhs[d_idx] = 0.0
        d_pk[d_idx] = d_p[d_idx]

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rhs, dt, VIJ, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        vdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_rhs[d_idx] += -Vj * vdotdwij / dt


class VelocityDivergenceSolid(Equation):
    def loop(self, d_idx, s_idx, s_m, s_rho, d_rhs, dt, d_u, d_v, d_w, s_ug,
             s_vg, s_wg, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        uij = d_u[d_idx] - s_ug[s_idx]
        vij = d_v[d_idx] - s_vg[s_idx]
        wij = d_w[d_idx] - s_wg[s_idx]
        vdotdwij = uij*DWIJ[0] + vij*DWIJ[1] + wij*DWIJ[2]
        d_rhs[d_idx] += -Vj * vdotdwij / dt


class DensityInvariance(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(DensityInvariance, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_pk):
        d_pk[d_idx] = d_p[d_idx]

    def post_loop(self, d_idx, d_rho, d_rhs, dt):
        rho0 = self.rho0
        d_rhs[d_idx] = (rho0 - d_rho[d_idx]) / (dt*dt*rho0)


class PressureCoeffMatrix(Equation):
    def initialize(self, d_idx, d_diag, d_odiag):
        d_diag[d_idx] = 0.0
        d_odiag[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_diag, d_odiag, s_pk, XIJ,
             DWIJ, R2IJ, EPS):
        rhoij = (s_rho[s_idx] + d_rho[d_idx])
        rhoij2_1 = 1.0/(rhoij*rhoij)

        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        fac = 8.0 * s_m[s_idx] * rhoij2_1 * xdotdwij / (R2IJ + EPS)

        if d_idx == s_idx:
            d_diag[d_idx] += fac
        d_odiag[d_idx] += -fac * s_pk[s_idx]


class PressureCoeffMatrixIterative(Equation):
    def initialize(self, d_idx, d_diag, d_odiag):
        d_diag[d_idx] = 0.0
        d_odiag[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_diag, d_odiag, s_pk, XIJ,
             DWIJ, R2IJ, EPS):
        rhoij = (s_rho[s_idx] + d_rho[d_idx])
        rhoij2_1 = 1.0/(rhoij*rhoij)

        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        fac = 8.0 * s_m[s_idx] * rhoij2_1 * xdotdwij / (R2IJ + EPS)

        d_diag[d_idx] += fac
        d_odiag[d_idx] += -fac * s_pk[s_idx]


class PPESolve(Equation):
    def __init__(self, dest, sources, rho0, omega=0.5, tolerance=0.05):
        self.rho0 = rho0
        self.conv = 0.0
        self.omega = omega
        self.tolerance = tolerance
        super(PPESolve, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_p, d_pk, d_rhs, d_odiag, d_diag, d_pdiff, d_V,
                  d_m):
        omega = self.omega
        rho = d_V[d_idx] * d_m[d_idx] / self.rho0
        if abs(d_diag[d_idx]) < 1e-9:
            # pnew = d_pk[d_idx]
            p = 0.0
        elif rho < 0.8:
            p = 0.0
        else:
            pnew = (d_rhs[d_idx] - d_odiag[d_idx]) / d_diag[d_idx]
            p = omega * pnew + (1.0 - omega) * d_pk[d_idx]

        d_pdiff[d_idx] = abs(p - d_pk[d_idx])
        d_p[d_idx] = p
        d_pk[d_idx] = p

    def reduce(self, dst, t, dt):
        pdiff = dst.pdiff.mean()
        pmean = numpy.abs(dst.p).mean()
        conv = pdiff/abs(pmean)
        self.conv = 1 if conv < self.tolerance else -1

    def converged(self):
        return self.conv


class UpdateGhostPressure(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_pk):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_pk[d_idx] = d_pk[idx]
            d_p[d_idx] = d_p[idx]


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au_pre, d_av_pre, d_aw_pre):
        d_au_pre[d_idx] = 0.0
        d_av_pre[d_idx] = 0.0
        d_aw_pre[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au_pre,
             d_av_pre, d_aw_pre, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        pji = (s_p[s_idx] - d_p[d_idx])
        fac = -Vj * pji / d_rho[d_idx]

        d_au_pre[d_idx] += fac * DWIJ[0]
        d_av_pre[d_idx] += fac * DWIJ[1]
        d_aw_pre[d_idx] += fac * DWIJ[2]


class SetPressureSolid(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0,
                 hg_correction=True):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.hg_correction = hg_correction
        super(SetPressureSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho, d_pk):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]
        if self.hg_correction:
            d_p[d_idx] = max(0.0, d_p[d_idx])
        d_pk[d_idx] = d_p[d_idx]


class ISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, nu, rho0, c0, alpha, beta=0.0,
                 gx=0.0, gy=0.0, gz=0.0, variant="CR", tolerance=0.05,
                 omega=0.5, hg_correction=True, has_ghosts=False):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.dim = dim
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.rho0 = rho0
        self.tolerance = tolerance
        self.omega = omega
        self.hg_correction = hg_correction
        self.has_ghosts = has_ghosts

    def add_user_options(self, group):
        group.add_argument(
            "--variant", action="store", dest="variant",
            type=str, choices=['DF', 'DI', 'DFDI'],
            help="ISPH variant (defaults to \"CR\" Cummins and Rudmann)."
        )
        group.add_argument(
            "--tol", action="store", dest="tolerance",
            type=float,
            help="Tolerance for convergence."
        )
        group.add_argument(
            "--omega", action="store", dest="omega",
            type=float,
            help="Omega for convergence."
        )
        group.add_argument(
            '--alpha', action='store', type=float, dest='alpha',
            default=None,
            help='Artificial viscosity.'
        )

    def consume_user_options(self, options):
        _vars = ['variant', 'tolerance', 'omega', 'alpha']
        data = dict((var, self._smart_getattr(options, var))
                    for var in _vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        import pysph.base.kernels as kern
        if kernel is None:
            kernel = kern.QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = ISPHStep
        if self.variant == "DI":
            step_cls = ISPHDIStep

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        if integrator_cls is not None:
            cls = integrator_cls
        else:
            cls = PECIntegrator

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def _get_velocity_bc(self):
        from pysph.sph.wc.transport_velocity import SetWallVelocity
        eqs = [SetWallVelocity(dest=s, sources=self.fluids)
               for s in self.solids]
        return Group(equations=eqs)

    def _get_pressure_bc(self):
        eqs = []
        for solid in self.solids:
            eqs.append(
                SetPressureSolid(
                    dest=solid, sources=self.fluids,
                    gx=self.gx, gy=self.gy, gz=self.gz,
                    hg_correction=self.hg_correction
                )
            )
        return Group(equations=eqs) if eqs else None

    def get_equations(self):
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.wc.transport_velocity import VolumeSummation

        all = self.fluids + self.solids

        eq1, stg1 = [], []
        if self.solids:
            g0 = self._get_velocity_bc()
            stg1.append(g0)

        for fluid in self.fluids:
            eq1.append(
                MomentumEquationViscosity(
                    dest=fluid, sources=all, nu=self.nu, alpha=self.alpha,
                    beta=self.beta, c0=self.c0, gx=self.gx, gy=self.gy,
                    gz=self.gz)
            )
        stg1.append(Group(equations=eq1))

        eq2, stg2 = [], []

        if self.solids:
            g0 = self._get_velocity_bc()
            stg2.append(g0)

        if self.variant == 'DI':
            for fluid in self.fluids:
                eq2.append(SummationDensity(dest=fluid, sources=all))
                eq2.append(VolumeSummation(dest=fluid, sources=all))
            stg2.append(Group(equations=eq2))

        eq2 = []
        for fluid in self.fluids:
            if self.variant == 'DI':
                eq2.append(
                    DensityInvariance(dest=fluid, sources=all, rho0=self.rho0)
                )
            else:
                eq2.append(VolumeSummation(dest=fluid, sources=all))
                eq2.append(VelocityDivergence(dest=fluid, sources=self.fluids))
                if self.solids:
                    eq2.append(VelocityDivergenceSolid(
                        dest=fluid, sources=self.solids
                    ))
        stg2.append(Group(equations=eq2))

        eq3 = []
        for fluid in self.fluids:
            eq3.append(PressureCoeffMatrixIterative(dest=fluid, sources=all))
            eq3.append(PPESolve(dest=fluid, sources=all, rho0=self.rho0,
                                tolerance=self.tolerance, omega=self.omega))
        eq3 = Group(equations=eq3)

        solver_eqns = []
        if self.has_ghosts:
            ghost_eqns = Group(
                equations=[UpdateGhostPressure(dest=fluid, sources=None)
                           for fluid in self.fluids],
                real=False
            )
            solver_eqns = [ghost_eqns]

        if self.solids:
            g3 = self._get_pressure_bc()
            solver_eqns.append(g3)

        solver_eqns.append(eq3)

        stg2.append(
            Group(
                equations=solver_eqns, iterate=True, max_iterations=100,
                min_iterations=2
            )
        )

        if self.has_ghosts:
            ghost_eqns = Group(
                equations=[UpdateGhostPressure(dest=fluid, sources=None)
                           for fluid in self.fluids],
                real=False
            )
            stg2.append(ghost_eqns)
        eq4 = []
        for fluid in self.fluids:
            eq4.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all)
            )

        if self.solids:
            g3 = self._get_pressure_bc()
            stg2.append(g3)

        if self.solids:
            g0 = self._get_velocity_bc()
            stg2.append(g0)

        stg2.append(Group(equations=eq4))

        return MultiStageEquations([stg1, stg2])

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_isph(name='junk',
                                        gid=particle_arrays['fluid'].gid)
        props = list(dummy.properties.keys())
        props += [dict(name=x, stride=v) for x, v in dummy.stride.items()]
        constants = [dict(name=x, data=v) for x, v in dummy.constants.items()]
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            for const in constants:
                pa.add_constant(**const)

        solid_props = ['wij', 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'pk']
        for solid in self.solids:
            pa = particle_arrays[solid]
            for prop in solid_props:
                pa.add_property(prop)
