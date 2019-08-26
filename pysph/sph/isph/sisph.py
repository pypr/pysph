"""
Simple iterative Incompressible SPH
See https://arxiv.org/abs/1908.01762 for details
"""
import numpy
from numpy import sqrt
from compyle.api import declare
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation, Group, MultiStageEquations


def get_particle_array_sisph(constants=None, **props):
    sisph_props = [
        'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0', 'diag', 'odiag',
        'pk', 'rhs', 'pdiff', 'wg', 'vf', 'vg', 'ug', 'wij', 'wf', 'uf',
        'V', 'au', 'av', 'aw', 'dt_force', 'dt_cfl', 'vmag',
        'auhat', 'avhat', 'awhat', 'p0', 'uhat', 'vhat', 'what',
        'uhat0', 'vhat0', 'what0', 'pabs'
    ]

    pa = get_particle_array(
        additional_props=sisph_props, constants=constants, **props
    )

    pa.add_constant('iters', [0.0])
    pa.add_constant('pmax', [0.0])
    pa.add_output_arrays(['p', 'V', 'vmag', 'p0'])
    return pa


class SISPHIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations(0)

        self.stage1()

        self.update_domain()

        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations(1, update_nnps=False)

        self.stage2()

        self.update_domain()

        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass


class SISPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v,
                   d_w, d_u0, d_v0, d_w0, dt):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_u0, d_v0, d_w0,
               d_x0, d_y0, d_z0, d_au, d_av, d_aw, d_vmag, d_dt_cfl,
               d_dt_force, dt):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + 0.5*dt * (d_u[d_idx] + d_u0[d_idx])
        d_y[d_idx] = d_y0[d_idx] + 0.5*dt * (d_v[d_idx] + d_v0[d_idx])
        d_z[d_idx] = d_z0[d_idx] + 0.5*dt * (d_w[d_idx] + d_w0[d_idx])

        d_vmag[d_idx] = sqrt(d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx] +
                             d_w[d_idx]*d_w[d_idx])

        d_dt_cfl[d_idx] = 2.0*d_vmag[d_idx]

        au = (d_u[d_idx] - d_u0[d_idx])/dt
        av = (d_v[d_idx] - d_v0[d_idx])/dt
        aw = (d_w[d_idx] - d_w0[d_idx])/dt

        d_dt_force[d_idx] = 4.0*(au*au + av*av + aw*aw)


class SISPHGTVFStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v,
                   d_w, d_u0, d_v0, d_w0, d_uhat, d_vhat, d_what, d_uhat0,
                   d_vhat0, d_what0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_uhat0[d_idx] = d_uhat[d_idx]
        d_vhat0[d_idx] = d_vhat[d_idx]
        d_what0[d_idx] = d_what[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_x0, d_y0, d_z0,
               d_au, d_av, d_aw, d_uhat, d_vhat, d_what, d_auhat, d_avhat,
               d_awhat, d_uhat0, d_vhat0, d_what0, d_vmag, d_dt_cfl, dt,
               d_u0, d_v0, d_w0, d_dt_force):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx] +
                             d_w[d_idx]*d_w[d_idx])
        d_dt_cfl[d_idx] = 2.0*d_vmag[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt*d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + 0.5*dt * (d_uhat[d_idx] + d_uhat0[d_idx])
        d_y[d_idx] = d_y0[d_idx] + 0.5*dt * (d_vhat[d_idx] + d_vhat0[d_idx])
        d_z[d_idx] = d_z0[d_idx] + 0.5*dt * (d_what[d_idx] + d_what0[d_idx])

        au = (d_u[d_idx] - d_u0[d_idx])/dt
        av = (d_v[d_idx] - d_v0[d_idx])/dt
        aw = (d_w[d_idx] - d_w0[d_idx])/dt

        d_dt_force[d_idx] = 4*(au*au + av*av + aw*aw)


class MomentumEquationBodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationBodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


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

    def post_loop(self, d_idx, d_rho, d_rhs, dt):
        rho0 = self.rho0
        d_rhs[d_idx] = (rho0 - d_rho[d_idx]) / (dt*dt*rho0)


class PressureCoeffMatrixIterative(Equation):
    def initialize(self, d_idx, d_diag, d_odiag):
        d_diag[d_idx] = 0.0
        d_odiag[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_diag, d_odiag, s_pk, XIJ,
             DWIJ, R2IJ, EPS):
        rhoij = (s_rho[s_idx] + d_rho[d_idx])
        rhoij2_1 = 1.0/(d_rho[d_idx]*rhoij)

        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        fac = 4.0 * s_m[s_idx] * rhoij2_1 * xdotdwij / (R2IJ + EPS)

        d_diag[d_idx] += fac
        d_odiag[d_idx] += -fac * s_pk[s_idx]


class PPESolve(Equation):
    def __init__(self, dest, sources, rho0, rho_cutoff=0.8, omega=0.5,
                 tolerance=0.05, max_iterations=1000):
        self.rho0 = rho0
        self.rho_cutoff = rho_cutoff
        self.conv = 0.0
        self.omega = omega
        self.tolerance = tolerance
        self.count = 0.0
        self.max_iterations = max_iterations
        super(PPESolve, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_p, d_pk, d_rhs, d_odiag, d_diag, d_pdiff,
                  d_rho, d_m, d_pabs, d_pmax):
        omega = self.omega
        rho = d_rho[d_idx] / self.rho0
        diag = d_diag[d_idx]
        if abs(diag) < 1e-12 or rho < self.rho_cutoff:
            p = 0.0
        else:
            pnew = (d_rhs[d_idx] - d_odiag[d_idx]) / diag
            p = omega * pnew + (1.0 - omega) * d_pk[d_idx]

        d_pdiff[d_idx] = abs(p - d_pk[d_idx])
        d_pabs[d_idx] = abs(p)
        d_p[d_idx] = p
        d_pk[d_idx] = p
        d_pmax[0] = max(abs(d_pmax[0]), d_p[d_idx])

    def reduce(self, dst, t, dt):
        self.count += 1
        dst.iters[0] = self.count
        if dst.gpu is not None:
            from compyle.array import sum
            n = dst.gpu.p.length
            pdiff = sum(dst.gpu.pdiff, dst.backend) / n
            pmean = sum(dst.gpu.pabs, dst.backend) / n
            conv = pdiff/pmean
            if pmean < 1.0:
                conv = pdiff
            self.conv = 1 if conv < self.tolerance else -1
        else:
            pdiff = dst.pdiff.mean()
            pmean = numpy.abs(dst.p).mean()
            conv = pdiff/pmean
            if pmean < 1.0:
                conv = pdiff
            self.conv = 1 if conv < self.tolerance else -1

    def converged(self):
        if self.conv == 1 and self.count < self.max_iterations:
            self.count = 0
        if self.count > self.max_iterations:
            self.count = 0
            print("Max iterations exceeded")
        return self.conv


class UpdateGhostPressure(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_pk):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_pk[d_idx] = d_pk[idx]
            d_p[d_idx] = d_p[idx]


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au,
             d_av, d_aw, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        pji = (s_p[s_idx] - d_p[d_idx])
        fac = -Vj * pji / d_rho[d_idx]

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]


class MomentumEquationPressureGradientSymmetric(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au, d_av, d_aw,
             DWIJ):
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        rhoj2 = s_rho[s_idx]*s_rho[s_idx]
        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2
        fac = -s_m[s_idx] * pij

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]


class EvaluateNumberDensity(Equation):
    def initialize(self, d_idx, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, WIJ):
        d_wij[d_idx] += WIJ


class VolumeSummationBand(Equation):
    def initialize(self, d_idx, d_rhoband):
        d_rhoband[d_idx] = 0.0

    def loop(self, d_idx, d_rhoband, d_m, WIJ):
        d_rhoband[d_idx] += WIJ * d_m[d_idx]


class SetPressureSolid(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0,
                 hg_correction=True):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.hg_correction = hg_correction
        super(SetPressureSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho, d_pk):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]
        if self.hg_correction:
            d_p[d_idx] = max(0.0, d_p[d_idx])
        d_pk[d_idx] = d_p[d_idx]


class GTVFAcceleration(Equation):
    def __init__(self, dest, sources, pref, internal_flow=False,
                 use_pref=False):
        self.pref = pref
        assert self.pref is not None, "pref should not be None"
        self.internal = internal_flow
        self.hij_fac = 1 if self.internal else 0.5
        self.use_pref = use_pref
        super(GTVFAcceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat, d_p0, d_p, d_pmax):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

        if self.internal:
            if self.use_pref:
                d_p0[d_idx] = self.pref
            else:
                pref = 2*d_pmax[0]
                d_p0[d_idx] = pref
        else:
            d_p0[d_idx] = min(10*abs(d_p[d_idx]), self.pref)

    def loop(self, d_p0, s_m, s_idx, d_rho, d_idx, d_auhat, d_avhat,
             d_awhat, XIJ, RIJ, SPH_KERNEL,
             HIJ):
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        tmp = -d_p0[d_idx] * s_m[s_idx]/rhoi2

        dwijhat = declare('matrix(3)')
        SPH_KERNEL.gradient(XIJ, RIJ, self.hij_fac*HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class SmoothedVelocity(Equation):
    def initialize(self, d_ax, d_ay, d_az, d_idx):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0
        d_az[d_idx] = 0.0

    def loop(self, d_ax, d_ay, d_az, d_idx, s_uhat, s_vhat, s_what, s_idx, s_m,
             s_rho, WIJ):
        fac = s_m[s_idx]*WIJ / s_rho[s_idx]
        d_ax[d_idx] += fac*s_uhat[s_idx]
        d_ay[d_idx] += fac*s_vhat[s_idx]
        d_az[d_idx] += fac*s_what[s_idx]


class SolidWallNoSlipBC(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, s_m,
             d_u, d_v, d_w,
             d_au, d_av, d_aw,
             s_ug, s_vg, s_wg,
             DWIJ, R2IJ, EPS, XIJ):
        mj = s_m[s_idx]
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        rhoij1 = 1.0/(rhoi + rhoj)

        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        tmp = mj * 4 * self.nu * rhoij1 * Fij/(R2IJ + EPS)

        d_au[d_idx] += tmp * (d_u[d_idx] - s_ug[s_idx])
        d_av[d_idx] += tmp * (d_v[d_idx] - s_vg[s_idx])
        d_aw[d_idx] += tmp * (d_w[d_idx] - s_wg[s_idx])


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


class SISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, nu, rho0, c0, alpha=0.0, beta=0.0,
                 gx=0.0, gy=0.0, gz=0.0, tolerance=0.05,
                 omega=0.5, hg_correction=False, has_ghosts=False,
                 pref=None, gtvf=False, symmetric=False, rho_cutoff=0.8,
                 max_iterations=1000, internal_flow=False,
                 use_pref=False):
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
        self.rho0 = rho0
        self.rho_cutoff = rho_cutoff
        self.tolerance = tolerance
        self.omega = omega
        self.hg_correction = hg_correction
        self.has_ghosts = has_ghosts
        self.pref = pref
        self.gtvf = gtvf
        self.symmetric = symmetric
        self.max_iterations = max_iterations
        self.internal_flow = internal_flow
        self.use_pref = use_pref

    def add_user_options(self, group):
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
        add_bool_argument(
            group, 'gtvf', dest='gtvf', default=None,
            help='Use GTVF.'
        )
        add_bool_argument(
            group, 'symmetric', dest='symmetric', default=None,
            help='Use symmetric form of pressure gradient.'
        )
        add_bool_argument(
            group, 'internal', dest='internal_flow', default=None,
            help='If the simulation is internal or external.'
        )

    def consume_user_options(self, options):
        _vars = ['tolerance', 'omega', 'alpha', 'gtvf', 'symmetric',
                 'internal_flow']
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

        step_cls = SISPHStep
        if self.gtvf:
            step_cls = SISPHGTVFStep

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        if integrator_cls is not None:
            cls = integrator_cls
        else:
            cls = SISPHIntegrator

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def _get_velocity_bc(self):
        from pysph.sph.isph.wall_normal import SetWallVelocityNew

        eqs = [SetWallVelocityNew(dest=s, sources=self.fluids)
               for s in self.solids]
        return Group(equations=eqs)

    def _get_pressure_bc(self):
        eqs = []
        all_solids = self.solids
        for solid in all_solids:
            eqs.append(
                EvaluateNumberDensity(
                    dest=solid, sources=self.fluids
                )
            )
            eqs.append(
                SetPressureSolid(
                    dest=solid, sources=self.fluids,
                    gx=self.gx, gy=self.gy, gz=self.gz,
                    hg_correction=self.hg_correction
                )
            )

        return Group(equations=eqs) if eqs else None

    def _get_normals(self, pa):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.isph.wall_normal import ComputeNormals, SmoothNormals

        pa.add_property('normal', stride=3)
        pa.add_property('normal_tmp', stride=3)

        name = pa.name

        seval = SPHEvaluator(
            arrays=[pa], equations=[
                Group(equations=[
                    ComputeNormals(dest=name, sources=[name])
                ]),
                Group(equations=[
                    SmoothNormals(dest=name, sources=[name])
                ]),
            ],
            dim=self.dim
        )
        seval.evaluate()

    def _get_viscous_eqns(self):
        from pysph.sph.wc.transport_velocity import (
            MomentumEquationArtificialViscosity)
        from pysph.sph.wc.viscosity import LaminarViscosity
        from pysph.sph.wc.gtvf import MomentumEquationArtificialStress

        all = self.fluids + self.solids

        eq, stg = [], []
        for fluid in self.fluids:
            eq.append(SummationDensity(dest=fluid, sources=all))
        stg.append(Group(equations=eq, real=False))

        eq = []
        for fluid in self.fluids:
            if self.nu > 0.0:
                eq.append(
                    LaminarViscosity(fluid, sources=self.fluids,
                                     nu=self.nu)
                )
            if self.alpha > 0.0:
                eq.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=self.fluids, c0=self.c0,
                        alpha=self.alpha
                    )
                )
            eq.append(
                MomentumEquationBodyForce(
                    fluid, sources=None, gx=self.gx, gy=self.gy,
                    gz=self.gz)
            )
            if self.gtvf:
                eq.append(
                    MomentumEquationArtificialStress(
                        dest=fluid, sources=self.fluids, dim=self.dim)
                )
            if self.solids and self.nu > 0.0:
                eq.append(
                    SolidWallNoSlipBC(
                        dest=fluid, sources=self.solids, nu=self.nu
                    )
                )
        stg.append(Group(equations=eq))
        return stg

    def _get_ppe(self):
        from pysph.sph.wc.transport_velocity import VolumeSummation

        all = self.fluids + self.solids
        all_solids = self.solids

        eq, stg = [], []

        for fluid in self.fluids:
            eq.append(SummationDensity(dest=fluid, sources=all))
        stg.append(Group(equations=eq, real=False))

        eq2 = []
        for fluid in self.fluids:
            eq2.append(VolumeSummation(dest=fluid, sources=all))
            eq2.append(
                VelocityDivergence(
                    dest=fluid,
                    sources=self.fluids
                ))
            if self.solids:
                eq2.append(
                    VelocityDivergenceSolid(fluid, sources=self.solids)
                )
        stg.append(Group(equations=eq2))

        solver_eqns = []
        if self.has_ghosts:
            ghost_eqns = Group(
                equations=[UpdateGhostPressure(dest=fluid, sources=None)
                           for fluid in self.fluids],
                real=False
            )
            solver_eqns = [ghost_eqns]

        if all_solids:
            g3 = self._get_pressure_bc()
            solver_eqns.append(g3)

        eq3 = []
        for fluid in self.fluids:
            if not fluid == 'outlet':
                eq3.append(
                    PressureCoeffMatrixIterative(dest=fluid, sources=all)
                )
                eq3.append(
                    PPESolve(
                        dest=fluid, sources=all, rho0=self.rho0,
                        rho_cutoff=self.rho_cutoff, tolerance=self.tolerance,
                        omega=self.omega, max_iterations=self.max_iterations
                    )
                )
        eq3 = Group(equations=eq3)

        solver_eqns.append(eq3)

        stg.append(
            Group(
                equations=solver_eqns, iterate=True,
                max_iterations=self.max_iterations, min_iterations=2
            )
        )

        if self.has_ghosts:
            ghost_eqns = Group(
                equations=[UpdateGhostPressure(dest=fluid, sources=None)
                           for fluid in self.fluids],
                real=False
            )
            stg.append(ghost_eqns)
        return stg

    def get_equations(self):
        all = self.fluids + self.solids
        all_solids = self.solids

        stg1 = []
        if all_solids:
            g0 = self._get_velocity_bc()
            stg1.append(g0)

        stg1.extend(self._get_viscous_eqns())

        stg2 = []
        if all_solids:
            g0 = self._get_velocity_bc()
            stg2.append(g0)

        stg2.extend(self._get_ppe())

        if all_solids:
            g3 = self._get_pressure_bc()
            stg2.append(g3)

        if all_solids:
            g0 = self._get_velocity_bc()
            stg2.append(g0)

        eq4 = []
        for fluid in self.fluids:
            if self.symmetric:
                eq4.append(
                    MomentumEquationPressureGradientSymmetric(fluid, all)
                )
            else:
                eq4.append(
                    MomentumEquationPressureGradient(fluid, sources=all)
                )
            if self.gtvf:
                eq4.append(
                    GTVFAcceleration(dest=fluid, sources=all, pref=self.pref,
                                     internal_flow=self.internal_flow,
                                     use_pref=self.use_pref)
                )
        stg2.append(Group(equations=eq4))
        return MultiStageEquations([stg1, stg2])

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_sisph(
            name='junk', gid=particle_arrays['fluid'].gid
        )
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

        solid_props = ['wij', 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 'pk', 'V']
        all_solids = self.solids
        for solid in all_solids:
            pa = particle_arrays[solid]
            for prop in solid_props:
                pa.add_property(prop)
            self._get_normals(pa)
            pa.add_output_arrays(['p', 'ug', 'vg', 'wg', 'normal'])
