"""The basic equations for the IISPH formulation of

    M. Ihmsen, J. Cornelis, B. Solenthaler, C. Horvath, M. Teschner, "Implicit
    Incompressible SPH," IEEE Transactions on Visualization and Computer
    Graphics, vol. 20, no. 3, pp. 426-435, March 2014.
    http://dx.doi.org/10.1109/TVCG.2013.105

"""

from numpy import sqrt, fabs
from compyle.api import declare
from pysph.base.particle_array import get_ghost_tag
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from pysph.sph.scheme import Scheme, add_bool_argument


GHOST_TAG = get_ghost_tag()


class IISPHStep(IntegratorStep):
    """A straightforward and simple integrator to be used for IISPH.
    """
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_uadv, d_vadv, d_wadv, d_au, d_av, d_aw,
               d_ax, d_ay, d_az, dt):
        d_u[d_idx] = d_uadv[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_vadv[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_wadv[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]


class NumberDensity(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


class SummationDensityBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(SummationDensityBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, s_idx, s_V, WIJ):
        d_rho[d_idx] += self.rho0/s_V[s_idx]*WIJ


class NormalizedSummationDensity(Equation):
    def initialize(self, d_idx, d_rho, d_rho_adv, d_rho0, d_V):
        d_rho0[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0
        d_rho_adv[d_idx] = 0.0
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_rho_adv, d_V, s_idx, s_m, s_rho0, WIJ):
        tmp = s_m[s_idx]*WIJ
        d_rho[d_idx] += tmp
        d_rho_adv[d_idx] += tmp/s_rho0[s_idx]
        d_V[d_idx] += WIJ

    def post_loop(self, d_idx, d_rho, d_rho_adv):
        d_rho[d_idx] /= d_rho_adv[d_idx]


class AdvectionAcceleration(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(AdvectionAcceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_uadv, d_vadv, d_wadv):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz
        d_uadv[d_idx] = 0.0
        d_vadv[d_idx] = 0.0
        d_wadv[d_idx] = 0.0

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_uadv, d_vadv, d_wadv,
                  d_u, d_v, d_w, dt=0.0):
        d_uadv[d_idx] = d_u[d_idx] + dt*d_au[d_idx]
        d_vadv[d_idx] = d_v[d_idx] + dt*d_av[d_idx]
        d_wadv[d_idx] = d_w[d_idx] + dt*d_aw[d_idx]


class ViscosityAcceleration(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(ViscosityAcceleration, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw, s_idx, s_m, EPS,
             VIJ, XIJ, RHOIJ1, R2IJ, DWIJ):
        dwijdotxij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        fac = 2.0*self.nu*s_m[s_idx]*RHOIJ1*dwijdotxij/(R2IJ + EPS)
        d_au[d_idx] += fac*VIJ[0]
        d_av[d_idx] += fac*VIJ[1]
        d_aw[d_idx] += fac*VIJ[2]


class ViscosityAccelerationBoundary(Equation):
    """The acceleration on the fluid due to a boundary.
    """
    def __init__(self, dest, sources, rho0, nu):
        self.nu = nu
        self.rho0 = rho0
        super(ViscosityAccelerationBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw, d_rho, s_idx, s_V, EPS,
             VIJ, XIJ, R2IJ, DWIJ):
        phi_b = self.rho0/(s_V[s_idx]*d_rho[d_idx])
        dwijdotxij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        fac = 2.0*self.nu*phi_b*dwijdotxij/(R2IJ + EPS)
        d_au[d_idx] += fac*VIJ[0]
        d_av[d_idx] += fac*VIJ[1]
        d_aw[d_idx] += fac*VIJ[2]


class ComputeDII(Equation):
    def initialize(self, d_idx, d_dii0, d_dii1, d_dii2):
        d_dii0[d_idx] = 0.0
        d_dii1[d_idx] = 0.0
        d_dii2[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_dii0, d_dii1, d_dii2,
             s_idx, s_m, DWIJ):
        rho_1 = 1.0/d_rho[d_idx]
        fac = -s_m[s_idx]*rho_1*rho_1
        d_dii0[d_idx] += fac*DWIJ[0]
        d_dii1[d_idx] += fac*DWIJ[1]
        d_dii2[d_idx] += fac*DWIJ[2]


class ComputeDIIBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeDIIBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_dii0, d_dii1, d_dii2, d_rho,
             s_idx, s_m, s_V, DWIJ):
        rhoi1 = 1.0/d_rho[d_idx]
        fac = -rhoi1*rhoi1*self.rho0/s_V[s_idx]
        d_dii0[d_idx] += fac*DWIJ[0]
        d_dii1[d_idx] += fac*DWIJ[1]
        d_dii2[d_idx] += fac*DWIJ[2]


class ComputeRhoAdvection(Equation):
    def initialize(self, d_idx, d_rho_adv, d_rho, d_p0, d_p, d_piter, d_aii):
        d_rho_adv[d_idx] = d_rho[d_idx]
        d_p0[d_idx] = d_p[d_idx]
        d_piter[d_idx] = 0.5*d_p[d_idx]

    def loop(self, d_idx, d_rho, d_rho_adv, d_uadv, d_vadv, d_wadv, d_u,
             d_v, d_w, s_idx, s_m, s_uadv, s_vadv, s_wadv, DWIJ, dt=0.0):

        vijdotdwij = (d_uadv[d_idx] - s_uadv[s_idx])*DWIJ[0] + \
                     (d_vadv[d_idx] - s_vadv[s_idx])*DWIJ[1] + \
                     (d_wadv[d_idx] - s_wadv[s_idx])*DWIJ[2]

        d_rho_adv[d_idx] += dt*s_m[s_idx]*vijdotdwij


class ComputeRhoBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeRhoBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_rho_adv, d_uadv, d_vadv, d_wadv,
             s_idx, s_u, s_v, s_w, s_V, WIJ, DWIJ, dt=0.0):
        phi_b = self.rho0/s_V[s_idx]

        vijdotdwij = (d_uadv[d_idx] - s_u[s_idx])*DWIJ[0] + \
                     (d_vadv[d_idx] - s_v[s_idx])*DWIJ[1] + \
                     (d_wadv[d_idx] - s_w[s_idx])*DWIJ[2]
        d_rho_adv[d_idx] += dt*phi_b*vijdotdwij


class ComputeAII(Equation):
    def initialize(self, d_idx, d_aii):
        d_aii[d_idx] = 0.0

    def loop(self, d_idx, d_aii, d_dii0, d_dii1, d_dii2, d_m, d_rho,
             s_idx, s_m, s_rho, DWIJ):
        rho1 = 1.0/d_rho[d_idx]
        fac = d_m[d_idx]*rho1*rho1
        # The following is m_j (d_ii - d_ji) . DWIJ
        # DWIJ = -DWJI
        dijdotdwij = (d_dii0[d_idx] - fac*DWIJ[0])*DWIJ[0] + \
                     (d_dii1[d_idx] - fac*DWIJ[1])*DWIJ[1] + \
                     (d_dii2[d_idx] - fac*DWIJ[2])*DWIJ[2]
        d_aii[d_idx] += s_m[s_idx]*dijdotdwij


class ComputeAIIBoundary(Equation):
    """ This is important and not really discussed in the original IISPH paper.
    """
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeAIIBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_aii, d_dii0, d_dii1, d_dii2, d_rho,
             s_idx, s_m, s_V, DWIJ):
        phi_b = self.rho0/s_V[s_idx]
        rho1 = 1.0/d_rho[d_idx]
        fac = d_m[d_idx]*rho1*rho1
        dijdotdwij = ((d_dii0[d_idx] - fac*DWIJ[0])*DWIJ[0] +
                      (d_dii1[d_idx] - fac*DWIJ[1])*DWIJ[1] +
                      (d_dii2[d_idx] - fac*DWIJ[2])*DWIJ[2])
        d_aii[d_idx] += phi_b*dijdotdwij


class ComputeDIJPJ(Equation):
    def initialize(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2):
        d_dijpj0[d_idx] = 0.0
        d_dijpj1[d_idx] = 0.0
        d_dijpj2[d_idx] = 0.0

    def loop(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_m, s_rho, s_piter, DWIJ):
        rho1 = 1.0/s_rho[s_idx]
        fac = -s_m[s_idx]*rho1*rho1*s_piter[s_idx]
        d_dijpj0[d_idx] += fac*DWIJ[0]
        d_dijpj1[d_idx] += fac*DWIJ[1]
        d_dijpj2[d_idx] += fac*DWIJ[2]


class UpdateGhostProps(Equation):
    def __init__(self, dest, sources=None):
        super(UpdateGhostProps, self).__init__(dest, sources)
        # We do this to ensure that the ghost tag is indeed 2.
        # If not the initialize method will never work.
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_tag, d_orig_idx, d_dijpj0, d_dijpj1,
                   d_dijpj2, d_dii0, d_dii1, d_dii2, d_piter):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_dijpj0[d_idx] = d_dijpj0[idx]
            d_dijpj1[d_idx] = d_dijpj1[idx]
            d_dijpj2[d_idx] = d_dijpj2[idx]
            d_dii0[d_idx] = d_dii0[idx]
            d_dii1[d_idx] = d_dii1[idx]
            d_dii2[d_idx] = d_dii2[idx]
            d_piter[d_idx] = d_piter[idx]


class PressureSolve(Equation):
    def __init__(self, dest, sources, rho0, omega=0.5,
                 tolerance=1e-2, debug=False):
        self.rho0 = rho0
        self.omega = omega
        self.compression = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(PressureSolve, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_compression):
        d_p[d_idx] = 0.0
        d_compression[d_idx] = 0.0

    def loop(self, d_idx, d_p, d_piter, d_rho, d_m, d_dijpj0, d_dijpj1,
             d_dijpj2, s_idx, s_m, s_dii0, s_dii1, s_dii2,
             s_piter, s_dijpj0, s_dijpj1, s_dijpj2, DWIJ):

        # Note that a good way to check this is to see that when
        # d_idx == s_idx the contribution is zero, as is expected.
        rho1 = 1.0/d_rho[d_idx]
        fac = d_m[d_idx]*rho1*rho1*d_piter[d_idx]
        djkpk0 = s_dijpj0[s_idx] - fac*DWIJ[0]
        djkpk1 = s_dijpj1[s_idx] - fac*DWIJ[1]
        djkpk2 = s_dijpj2[s_idx] - fac*DWIJ[2]

        tmp0 = d_dijpj0[d_idx] - s_dii0[s_idx]*s_piter[s_idx] - djkpk0
        tmp1 = d_dijpj1[d_idx] - s_dii1[s_idx]*s_piter[s_idx] - djkpk1
        tmp2 = d_dijpj2[d_idx] - s_dii2[s_idx]*s_piter[s_idx] - djkpk2
        tmpdotdwij = tmp0*DWIJ[0] + tmp1*DWIJ[1] + tmp2*DWIJ[2]

        # This is corrected in the post_loop.
        d_p[d_idx] += s_m[s_idx]*tmpdotdwij

    def post_loop(self, d_idx, d_piter, d_p0, d_p, d_aii, d_rho_adv, d_rho,
                  d_compression, dt=0.0):
        dt2 = dt*dt
        # Recall that d_p now has \sum_{j\neq i} a_ij p_j
        tmp = self.rho0 - d_rho_adv[d_idx] - d_p[d_idx]*dt2
        dnr = d_aii[d_idx]*dt2

        if fabs(dnr) > 1e-9:
            # Clamp pressure to positive values.
            p = max((1.0 - self.omega)*d_piter[d_idx] +
                    self.omega/dnr*tmp, 0.0)
        else:
            p = 0.0

        if p != 0.0:
            d_compression[d_idx] = fabs(p*dnr - tmp) + self.rho0
        else:
            d_compression[d_idx] = self.rho0

        d_piter[d_idx] = p
        d_p[d_idx] = p

    def reduce(self, dst, t, dt):
        dst.tmp_comp[0] = serial_reduce_array(dst.compression > 0.0, 'sum')
        dst.tmp_comp[1] = serial_reduce_array(dst.compression, 'sum')
        dst.tmp_comp[:] = parallel_reduce_array(dst.tmp_comp, 'sum')
        if dst.tmp_comp[0] > 0:
            avg_rho = dst.tmp_comp[1]/dst.tmp_comp[0]
        else:
            avg_rho = self.rho0
        self.compression = fabs(avg_rho - self.rho0)/self.rho0

    def converged(self):
        debug = self.debug
        compression = self.compression

        if compression > self.tolerance:
            if debug:
                print("Not converged:", compression)
            return -1.0
        else:
            if debug:
                print("Converged:", compression)
            return 1.0


class PressureSolveBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureSolveBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_V, DWIJ):
        phi_b = self.rho0/s_V[s_idx]
        dijdotwij = (d_dijpj0[d_idx]*DWIJ[0] +
                     d_dijpj1[d_idx]*DWIJ[1] +
                     d_dijpj2[d_idx]*DWIJ[2])
        d_p[d_idx] += phi_b*dijdotwij


class UpdateGhostPressure(Equation):
    def initialize(self, d_idx, d_tag, d_orig_idx, d_p, d_piter):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_piter[d_idx] = d_piter[idx]
            d_p[d_idx] = d_p[idx]


class PressureForce(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_p, d_au, d_av, d_aw,
             s_idx, s_m, s_rho, s_p, DWIJ):
        rhoi1 = 1.0/d_rho[d_idx]
        rhoj1 = 1.0/s_rho[s_idx]
        fac = -s_m[s_idx]*(d_p[d_idx]*rhoi1*rhoi1 + s_p[s_idx]*rhoj1*rhoj1)
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw,
                  d_uadv, d_vadv, d_wadv, d_dt_cfl, d_dt_force):
        fac = d_au[d_idx]*d_au[d_idx] + d_av[d_idx]*d_av[d_idx] +\
                   d_aw[d_idx]*d_aw[d_idx]
        vmag = sqrt(d_uadv[d_idx]*d_uadv[d_idx] + d_vadv[d_idx]*d_vadv[d_idx] +
                    d_wadv[d_idx]*d_wadv[d_idx])
        d_dt_cfl[d_idx] = 2.0*vmag
        d_dt_force[d_idx] = 2.0*fac


class PressureForceBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureForceBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_au, d_av, d_aw,  d_p, s_idx, s_V, DWIJ):
        rho1 = 1.0/d_rho[d_idx]
        fac = -d_p[d_idx]*rho1*rho1*self.rho0/s_V[s_idx]
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]


class IISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, nu=0.0,
                 gx=0.0, gy=0.0, gz=0.0, omega=0.5, tolerance=1e-2,
                 debug=False, has_ghosts=False):
        '''The IISPH scheme

        Parameters
        ----------

        fluids : list(str)
            List of names of fluid particle arrays.
        solids : list(str)
            List of names of solid particle arrays.
        dim: int
            Dimensionality of the problem.
        rho0 : float
            Density of fluid.
        nu : float
            Kinematic viscosity.
        gx, gy, gz : float
            Componenents of body acceleration (gravity, external forcing etc.)
        omega : float
            Relaxation parameter for relaxed-Jacobi iterations.
        tolerance: float
            Tolerance for the convergence of pressure iterations as a fraction.
        debug: bool
            Produce some debugging output on iterations.
        has_ghosts: bool
            The problem has ghost particles so add equations for those.
        '''
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.omega = omega
        self.tolerance = tolerance
        self.debug = debug
        self.has_ghosts = has_ghosts

    def add_user_options(self, group):
        group.add_argument(
            '--omega', action="store", type=float, dest="omega",
            default=None, help="Relaxation parameter for Jacobi iterations."
        )
        group.add_argument(
            '--tolerance', action='store', type=float, dest='tolerance',
            default=None,
            help='Tolerance for convergence of iterations as a fraction'
        )
        add_bool_argument(
            group, 'iisph-debug', dest='debug', default=None,
            help="Produce some debugging output on convergence of iterations."
        )

    def consume_user_options(self, options):
        vars = ['omega', 'tolerance', 'debug']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """Configure the solver to be generated.

        This is to be called before `get_solver` is called.

        Parameters
        ----------

        dim : int
            Number of dimensions.
        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EulerIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = IISPHStep()

        cls = integrator_cls if integrator_cls is not None else EulerIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        has_ghosts = self.has_ghosts
        equations = []
        if self.solids:
            g1 = Group(
                equations=[NumberDensity(dest=x, sources=[x])
                           for x in self.solids]
            )
            equations.append(g1)

        g2 = Group(
            equations=[SummationDensity(dest=x, sources=self.fluids)
                       for x in self.fluids],
            real=False
        )
        equations.append(g2)

        if self.solids:
            g3 = Group(
                equations=[
                    SummationDensityBoundary(
                        dest=x, sources=self.solids, rho0=self.rho0
                    )
                    for x in self.fluids],
                real=False
            )
            equations.append(g3)

        eq = []
        for fluid in self.fluids:
            eq.extend([
                AdvectionAcceleration(
                    dest=fluid, sources=None,
                    gx=self.gx, gy=self.gy, gz=self.gz
                ),
                ComputeDII(dest=fluid, sources=self.fluids)
            ])
            if self.nu > 0.0:
                eq.append(
                    ViscosityAcceleration(
                        dest=fluid, sources=self.fluids, nu=self.nu
                    )
                )

            if self.solids:
                if self.nu > 0.0:
                    eq.append(
                        ViscosityAccelerationBoundary(
                            dest=fluid, sources=self.solids, nu=self.nu,
                            rho0=self.rho0,
                        )
                    )
                eq.append(
                    ComputeDIIBoundary(dest=fluid, sources=self.solids,
                                       rho0=self.rho0)
                )

        g4 = Group(equations=eq, real=False)
        equations.append(g4)

        eq = []
        for fluid in self.fluids:
            eq.extend([
                ComputeRhoAdvection(dest=fluid, sources=self.fluids),
                ComputeAII(dest=fluid, sources=self.fluids),
            ])
            if self.solids:
                eq.extend([
                    ComputeRhoBoundary(dest=fluid, sources=self.solids,
                                       rho0=self.rho0),
                    ComputeAIIBoundary(dest=fluid, sources=self.solids,
                                       rho0=self.rho0),
                ])
        g5 = Group(equations=eq)
        equations.append(g5)

        sg1 = Group(
            equations=[
                ComputeDIJPJ(dest=x, sources=self.fluids)
                for x in self.fluids
            ]
        )
        eq = []
        for fluid in self.fluids:
            eq.append(
                PressureSolve(dest=fluid, sources=self.fluids,
                              rho0=self.rho0, tolerance=self.tolerance,
                              debug=self.debug)
            )
            if self.solids:
                eq.append(
                    PressureSolveBoundary(
                        dest=fluid, sources=self.solids, rho0=self.rho0,
                    )
                )
        sg2 = Group(equations=eq)

        if has_ghosts:
            ghost1 = Group(
                equations=[UpdateGhostProps(dest=x, sources=None)
                           for x in self.fluids],
                real=False
            )
            ghost2 = Group(
                equations=[UpdateGhostPressure(dest=x, sources=None)
                           for x in self.fluids],
                real=False
            )
            solver_eqs = [sg1, ghost1, sg2, ghost2]
        else:
            solver_eqs = [sg1, sg2]
        g6 = Group(
            equations=solver_eqs,
            iterate=True, max_iterations=30, min_iterations=2
        )
        equations.append(g6)

        eq = []
        for fluid in self.fluids:
            eq.append(
                PressureForce(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                eq.append(
                    PressureForceBoundary(
                        dest=fluid, sources=self.solids, rho0=self.rho0
                    )
                )
        g7 = Group(equations=eq)
        equations.append(g7)

        return equations

    def setup_properties(self, particles, clean=True):
        """Setup the particle arrays so they have the right set of properties
        for this scheme.

        Parameters
        ----------

        particles : list
            List of particle arrays.

        clean : bool
            If True, removes any unnecessary properties.
        """
        from pysph.base.utils import get_particle_array_iisph
        dummy = get_particle_array_iisph()
        props = set(dummy.properties.keys())
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            for c, v in dummy.constants.items():
                if c not in pa.constants:
                    pa.add_constant(c, v)
            pa.set_output_arrays(dummy.output_property_arrays)

        if self.has_ghosts:
            particle_arrays = dict([(p.name, p) for p in particles])
            for fluid in self.fluids:
                pa = particle_arrays[fluid]
                pa.add_property('orig_idx', type='int')
