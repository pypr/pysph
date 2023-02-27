"""
References
-----------
    .. [Rosswog2009] Rosswog, Stephan. "Astrophysical smooth particle
        hydrodynamics." New Astronomy Reviews 53, no. 4-6 (2009): 78-104.
        https://doi.org/10.1016/j.newar.2009.08.007

    .. [Rosswog2015] Rosswog, Stephan. "Boosting the accuracy of SPH
        techniques: Newtonian and special-relativistic tests." Monthly
        Notices of the Royal Astronomical Society 448, no. 4 (2015):
        3628-3664. https://doi.org/10.1093/mnras/stv225.

    .. [Rosswog2020a] Rosswog, Stephan. "A simple, entropy-based dissipation
        trigger for SPH." The Astrophysical Journal 898, no. 1 (2020): 60.
        https://doi.org/10.3847/1538-4357/ab9a2e.

    .. [Rosswog2020b] Rosswog, Stephan. "The Lagrangian hydrodynamics code
        MAGMA2." Monthly Notices of the Royal Astronomical Society 498, no. 3
        (2020): 4230-4255. https://doi.org/10.1093/mnras/staa2591.

"""
from math import log

from compyle.types import declare, annotate
from pysph.base.particle_array import get_ghost_tag

from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.sph.wc.linalg import (augmented_matrix, gj_solve, identity,
                                 mat_mult, mat_vec_mult, dot)

GHOST_TAG = get_ghost_tag()


class MAGMA2Scheme(Scheme):
    """
    MAGMA2 formulations.

    Set of Equations: [Rosswog2020b]_

    Dissipation Limiter: [Rosswog2020a]_
    """

    def __init__(self, fluids, solids, dim, gamma, hfact=None, fkern=1.0,
                 adaptive_h_scheme='magma2', max_density_iterations=250,
                 density_iteration_tolerance=1e-3, alphamax=1.0, alphamin=0.1,
                 alphac=0.05, beta=2.0, eps=0.01, eta_crit=0.3, eta_fold=0.2,
                 ndes=None, reconstruction_order=2, formulation='mi1',
                 recycle_accelerations=True, has_ghosts=False, l0=log(1e-4),
                 l1=log(5e-2)):
        """
        Parameters
        ----------
        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries), currently
            not supported
        dim: int
            Dimensionality of the problem.
        gamma: float
            :math:`\\gamma` for Equation of state.
        hfact: float
            :math:`h_{fact}` for smoothing length adaptivity, also referred to
            as kernel_factor in other schemes like AKDE, MPM, GSPH.
        formulation: str, optional
            Set of governing equations for momentum and energy. Should be one
            of {'stdgrad', 'mi1', 'mi2'}, by default 'mi1'.
        adaptive_h_scheme: str, optional
            Procedure to adapt smoothing lengths. Should be one of
            {'gadget2', 'mpm'}, by default 'gadget2'.
        max_density_iterations: int, optional
            Maximum number of iterations to run for one density step if using
            MPM procedure to adapt smoothing lengths, by default 2.0
        density_iteration_tolerance: float, optional
            Maximum difference allowed in two successive density iterations
            if using MPM procedure to adapt smoothing lengths, by default 1e-3.
        alphamax : float, optional
            :math:`\\alpha_{max}` for artificial viscosity switch, by default
            1.0
        alphamin : float, optional
            :math:`\\alpha_{0}` for artificial viscosity switch, by default
            0.1
        alphac : float, optional
            :math:`\\alpha_{u}` for artificial conductivity, by default
            0.05
        beta : float, optional
            :math:`\\beta` for artificial viscosity, by default 2.0
        eps : float, optional
            Numerical parameter often used in denominator to avoid division
            by zero, by default 0.01
        eta_crit : float, optional
            :math:`\\eta_{crit}` for slope limiter, by default None
        eta_fold : float, optional
            :math:`\\eta_{fold}` for slope limiter, by default 0.2
        fkern : float, optional
            :math:`f_{kern}`, Factor to scale smoothing length for equivalence
            when using kernel with altered `radius_scale`, by default 1.0.
        ndes : int, optional
            :math:`n_{des}`, Desired number of neighbours to be in the kernel
            support of each particle, by default 300 for 3D.
        reconstruction_order : int, optional
            Order of reconstruction, by default 2.
        recycle_accelerations : bool, optional
            Weather to recycle accelerations, i.e., weather the accelerations
            used in the correction step can be reused in the successive
            prediction step, by default True.
        has_ghosts : bool, optional
            If ghost particles (either mirror or periodic) is used, by default
            False.
        l0 : float, optional
            Low entropy threshold parameter for dissipation trigger, by default
            log(1e-4).
        l1 : float, optional
            High entropy threshold parameter for dissipation trigger, by
            default log(5e-2).
        """
        self.h_scheme_choices = {'magma2', 'mpm'}
        self.formulation_choices = {'mi1', 'mi2', 'stdgrad'}
        self.reconstruction_order_choices = {0, 1, 2}
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.beta = beta
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts
        self.fkern = fkern
        self.alphamax = alphamax
        self.alphamin = alphamin
        self.alphac = alphac
        self.eta_crit = eta_crit
        self.eta_fold = eta_fold
        self.eps = eps
        self.recycle_accelerations = recycle_accelerations
        self.ndes = ndes
        self.adaptive_h_scheme = adaptive_h_scheme
        self.formulation = formulation
        self.reconstruction_order = reconstruction_order
        self.hfact = hfact
        self.ndes = ndes
        self.l0 = l0
        self.l1 = l1

    def add_user_options(self, group):
        group.add_argument("--adaptive-h", action="store",
                           dest="adaptive_h_scheme", default=None,
                           choices=self.h_scheme_choices,
                           help="Specify scheme for adaptive smoothing "
                                "lengths: %s" % self.h_scheme_choices)

        group.add_argument("--h-fact", action="store", type=float,
                           dest="hfact", default=None,
                           help="h_fact for smoothing length adaptivity.")

        group.add_argument("--formulation", action="store", dest="formulation",
                           default=None, choices=self.formulation_choices,
                           help="Specify the set of governing equations for "
                                "momentum and energy: "
                                "%s" % self.formulation_choices)

        group.add_argument("--reconstruction-order", action="store",
                           dest="reconstruction_order", type=int, default=None,
                           choices=self.reconstruction_order_choices,
                           help="Specify the order for reconstruction of "
                                "velocity and internal energy: "
                                "%s" % self.reconstruction_order_choices)

        group.add_argument("--alpha-max", action="store", type=float,
                           dest="alphamax", default=None,
                           help="alpha_max for the artificial viscosity "
                                "switch.")

        group.add_argument("--alpha-min", action="store", type=float,
                           dest="alphamin", default=None,
                           help="alpha_0 for the artificial viscosity "
                                "switch.")

        group.add_argument("--l0", action="store", type=float, dest="l0",
                           default=None,
                           help="Low entropy threshold parameter for "
                                "dissipation trigger.")

        group.add_argument("--l1", action="store", type=float, dest="l1",
                           default=None,
                           help="High entropy threshold parameter for "
                                "dissipation trigger.")

        group.add_argument("--beta", action="store", type=float, dest="beta",
                           default=None,
                           help="beta for the artificial viscosity.")

        group.add_argument("--gamma", action="store", type=float, dest="gamma",
                           default=None, help="gamma for the state equation.")

        group.add_argument("--n-des", action="store", type=float, dest="ndes",
                           default=None, help="Desired Number of neighbours to"
                                              " be in the kernel support of "
                                              "each particle.")

        add_bool_argument(group, 'recycle-accelerations',
                          dest='recycle_accelerations', default=None,
                          help="Reuse accelerations used in the correction "
                               "step in the successive prediction step.")

    def consume_user_options(self, options):
        vars = ['gamma', 'alphamax', 'beta', 'adaptive_h_scheme', 'ndes',
                'recycle_accelerations', 'formulation', 'hfact',
                'reconstruction_order', 'alphamin', 'l0', 'l1']
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        if kernel is None:
            if self.dim == 1:
                from pysph.base.kernels import WendlandQuinticC6_1D
                kernel = WendlandQuinticC6_1D(dim=self.dim)
            else:
                from pysph.base.kernels import WendlandQuinticC6
                kernel = WendlandQuinticC6(dim=self.dim)

        if hasattr(kernel, 'fkern'):
            self.fkern = kernel.fkern
        else:
            self.fkern = 1.0

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        if integrator_cls is not None:
            int_cls = integrator_cls
        else:
            if self.recycle_accelerations:
                int_cls = TVDRK2IntegratorWithRecycling
            else:
                int_cls = TVDRK2Integrator

        step_cls = TVDRK2Step
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = int_cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group

        all_pa = self.fluids + self.solids

        if self.reconstruction_order not in self.reconstruction_order_choices:
            raise ValueError("reconstruction_order must be one of: "
                             "%r." % self.reconstruction_order_choices)

        equations = []
        if self.adaptive_h_scheme == "magma2":
            if self.ndes is None:
                raise ValueError("ndes should be specified if smoothing "
                                 "lengths are to be adapted by MAGMA2 "
                                 "procedure.")
            else:
                g1p0 = []
                for fluid in self.fluids:
                    g1p0.append(IncreaseSmoothingLength(dest=fluid,
                                                        sources=None))
                equations.append(Group(equations=g1p0))

                g1p1 = []
                for fluid in self.fluids:
                    g1p1.append(UpdateSmoothingLength(dest=fluid,
                                                      sources=all_pa,
                                                      ndes=self.ndes))
                equations.append(Group(equations=g1p1))

                g2 = []
                from pysph.sph.basic_equations import SummationDensity
                for fluid in self.fluids:
                    g2.append(SummationDensity(dest=fluid, sources=all_pa))
                    g2.append(IdealGasEOS(dest=fluid, sources=None,
                                          gamma=self.gamma))
                    if self.reconstruction_order > 1:
                        g2.append(AuxiliaryGradient(dest=fluid, sources=all_pa,
                                                    dim=self.dim))
                equations.append(Group(equations=g2))

        elif self.adaptive_h_scheme == "mpm":
            if self.hfact is None:
                raise ValueError("hfact should be specified if smoothing "
                                 "lengths are to be adapted by MPM procedure.")
            else:
                g1 = []
                for fluid in self.fluids:
                    g1.append(SummationDensityMPMStyle(
                        dest=fluid, sources=all_pa, hfact=self.hfact,
                        density_iterations=True, dim=self.dim,
                        htol=self.density_iteration_tolerance))
                    equations.append(
                        Group(equations=g1, update_nnps=True, iterate=True,
                              max_iterations=self.max_density_iterations))

                g2 = []
                for fluid in self.fluids:
                    g2.append(IdealGasEOS(dest=fluid, sources=None,
                                          gamma=self.gamma))
                    if self.reconstruction_order > 1:
                        g2.append(AuxiliaryGradient(dest=fluid, sources=all_pa,
                                                    dim=self.dim))
                equations.append(Group(equations=g2))
        else:
            raise ValueError("adaptive_h_scheme must be one of: "
                             "%r." % self.h_scheme_choices)

        g3p1 = []
        for fluid in self.fluids:
            g3p1.append(CorrectionMatrix(dest=fluid, sources=all_pa,
                                         dim=self.dim))
        equations.append(Group(equations=g3p1))

        g3p2 = []
        for fluid in self.fluids:
            if self.reconstruction_order > 0:
                g3p2.append(FirstGradient(dest=fluid, sources=all_pa,
                                          dim=self.dim))
            if self.reconstruction_order > 1:
                g3p2.append(SecondGradient(dest=fluid, sources=all_pa,
                                           dim=self.dim))
            g3p2.append(EntropyBasedDissipationTrigger(dest=fluid,
                                                       sources=None,
                                                       alphamax=self.alphamax,
                                                       alphamin=self.alphamin,
                                                       fkern=self.fkern,
                                                       l0=self.l0,
                                                       l1=self.l1,
                                                       gamma=self.gamma))
        equations.append(Group(equations=g3p2))

        g4 = []
        for solid in self.solids:
            g4.append(WallBoundary(solid, sources=self.fluids, dim=self.dim))
        equations.append(Group(equations=g4))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(UpdateGhostProps(dest=fluid, sources=None,
                                           dim=self.dim))
            equations.append(Group(equations=gh, real=False))

        g5 = []
        for fluid in self.fluids:
            if self.formulation == 'mi1':
                g5.append(MomentumAndEnergyMI1(dest=fluid, sources=all_pa,
                                               dim=self.dim, beta=self.beta,
                                               fkern=self.fkern,
                                               eta_crit=self.eta_crit,
                                               eta_fold=self.eta_fold,
                                               alphac=self.alphac,
                                               eps=self.eps))
            elif self.formulation == 'mi2':
                g5.append(MomentumAndEnergyMI2(dest=fluid, sources=all_pa,
                                               dim=self.dim, beta=self.beta,
                                               fkern=self.fkern,
                                               eta_crit=self.eta_crit,
                                               eta_fold=self.eta_fold,
                                               alphac=self.alphac,
                                               eps=self.eps))
            elif self.formulation == 'stdgrad':
                g5.append(MomentumAndEnergyStdGrad(dest=fluid, sources=all_pa,
                                                   dim=self.dim,
                                                   beta=self.beta,
                                                   fkern=self.fkern,
                                                   eta_crit=self.eta_crit,
                                                   eta_fold=self.eta_fold,
                                                   alphac=self.alphac,
                                                   eps=self.eps))
            else:
                raise ValueError("formulation must be one of: "
                                 "%r." % self.formulation_choices)
            g5.append(
                EvaluateTildeMu(dest=fluid, sources=all_pa, dim=self.dim))
        equations.append(Group(equations=g5))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])

        props = ['rho', 'm', 'x', 'y', 'z', 'u', 'v', 'w', 'h', 'cs', 'p', 'e',
                 'au', 'av', 'aw', 'ae', 'pid', 'gid', 'tag', 'dwdh',
                 'converged', 'ah', 'arho', 'dt_cfl', 'u0', 'v0', 'w0']
        more_props = ['n', 'dndh', 'prevn', 'prevdndh', 'divv', 'an', 'h0',
                      'aalpha', 'tilmu', 'dt_adapt', 'aalpha0', 'ae0', 'ah0',
                      'an0', 'arho0', 'au0', 'av0', 'aw0']
        props.extend(more_props)
        output_props = 'm rho p u v w x y z e n divv h alpha'.split(' ')
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            # The initial smoothing length needs to be huge if using
            # adaptive_h_scheme='magma2'
            if self.adaptive_h_scheme == 'magma2':
                pa.h = 2.0 * pa.h
            pa.add_property('orig_idx', type='int')
            # Guess for number density.
            pa.add_property('n', data=pa.rho / pa.m)
            pa.add_property('s', data=pa.p / (pa.rho ** self.gamma))
            pa.add_property('alpha', data=self.alphamin)
            pa.add_property('dv', stride=9, data=0.0)
            pa.add_property('dvaux', stride=9, data=0.0)
            pa.add_property('invdm', stride=9, data=0.0)
            pa.add_property('cm', stride=9, data=0.0)
            pa.add_property('ddv', stride=27, data=0.0)
            pa.add_property('de', stride=3, data=0.0)
            pa.add_property('dde', stride=9, data=0.0)
            pa.add_property('deaux', stride=3, data=0.0)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('wij htmp alpha rho0'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)
            pa.add_property('cm', stride=9, data=0.0)
            pa.add_property('ddv', stride=27, data=0.0)
            pa.add_property('de', stride=3, data=0.0)
            pa.add_property('dde', stride=9, data=0.0)
            pa.add_property('dv', stride=9, data=0.0)
            pa.add_property('dvaux', stride=9, data=0.0)
            pa.add_property('deaux', stride=3, data=0.0)


class IncreaseSmoothingLength(Equation):
    """
    Increase smoothing length by 10%.
    """

    def initialize(self, d_idx, d_h):
        d_h[d_idx] *= 1.10


class UpdateSmoothingLength(Equation):
    """
    Sorts neighbours based on distance and uses the distance of nearest
    :math:`(n_{des} + 1)^{th}` particle to set the smoothing length. Here,
    :math:`n_{des}` is the desired number of neighbours to be in the kernel
    support of each particle.
    """

    def __init__(self, dest, sources, ndes):
        self.ndes = int(ndes)
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [quicksort]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, s_x, s_y, s_z, NBRS, N_NBRS,
                 SPH_KERNEL):
        i, ndes = declare('int', 2)
        ndes = self.ndes
        s_idx = declare('long')
        xij = declare('matrix(3)')
        rij = declare('matrix(1000)')  # May set a bigger or a smaller size.
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]
            rij[i] = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
        quicksort(rij, 0, N_NBRS - 1)

        if N_NBRS > 1:
            # Scheme recommends using (ndes + 1)th rij. The min() used below
            # is just an extra precaution. Btw, index of (ndes + 1)th element
            # is ndes as indexing starts from 0.
            i = min(ndes, N_NBRS - 1)
            d_h[d_idx] = rij[i] / SPH_KERNEL.radius_scale


class SummationDensityMPMStyle(Equation):
    """
    :class:`SummationDensity
    <pysph.sph.gas_dynamics.basic.SummationDensity>` modified to use
    number density and without grad-h terms.
    """

    def __init__(self, dest, sources, dim, density_iterations=False,
                 iterate_only_once=False, hfact=1.2, htol=1e-6):
        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.hfact = hfact
        self.htol = htol
        self.equation_has_converged = 1

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_n, d_dndh, d_prevn,
                   d_prevdndh, d_an):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]

        d_n[d_idx] = 0.0
        d_an[d_idx] = 0.0
        d_dndh[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, s_m, VIJ, WI, DWI, GHI, d_n,
             d_dndh, d_an):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        d_rho[d_idx] += mj * WI

        d_arho[d_idx] += mj * vijdotdwij
        d_an[d_idx] += vijdotdwij

        # gradient of kernel w.r.t h
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_h0, d_h, d_ah, d_converged, d_n, d_dndh,
                  d_an):
        # iteratively find smoothing length
        if self.density_iterations:
            if not (d_converged[d_idx] == 1):
                hi = d_h[d_idx]
                hi0 = d_h0[d_idx]

                # estimated, without summations
                ni = (self.hfact / hi) ** self.dim
                dndhi = - self.dim * d_n[d_idx] / hi

                # the non-linear function and it's derivative
                func = d_n[d_idx] - ni
                dfdh = d_dndh[d_idx] - dndhi

                # Newton Raphson estimate for the new h
                hnew = hi - func / dfdh

                # Nanny control for h
                if hnew > 1.2 * hi:
                    hnew = 1.2 * hi
                elif hnew < 0.8 * hi:
                    hnew = 0.8 * hi

                # check for convergence
                diff = abs(hnew - hi) / hi0

                if not ((diff < self.htol) or self.iterate_only_once):
                    # this particle hasn't converged. This means the
                    # entire group must be repeated until this fellow
                    # has converged, or till the maximum iteration has
                    # been reached.
                    self.equation_has_converged = -1

                    # set particle properties for the next
                    # iteration. For the 'converged' array, a value of
                    # 0 indicates the particle hasn't converged
                    d_h[d_idx] = hnew
                    d_converged[d_idx] = 0
                else:
                    d_ah[d_idx] = d_an[d_idx] / dndhi
                    d_converged[d_idx] = 1

    def converged(self):
        return self.equation_has_converged


class IdealGasEOS(Equation):
    """
    :class:`IdealGasEOS
    <pysph.sph.gas_dynamics.basic.IdealGasEOS>` modified to avoid repeated
    calculations using :meth:`loop() <pysph.sph.equation.Equation.loop()>`.
    Doing the same using :meth:`post_loop()
    <pysph.sph.equation.Equation.loop()>`.
    """

    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super(IdealGasEOS, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]
        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])


class AuxiliaryGradient(Equation):
    """
    Auxiliary first gradient calculated using analytical gradient of kernel
    and without using density.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult, augmented_matrix, identity, gj_solve, mat_vec_mult]

    def initialize(self, d_dvaux, d_idx, d_invdm, d_deaux):
        dsi2, i, dim, dimsq = declare('int', 4)
        dimsq = self.dimsq
        dim = self.dim
        dsi2 = dimsq * d_idx
        for i in range(dim):
            d_deaux[dim * d_idx + i] = 0.0

        for i in range(dimsq):
            d_dvaux[dsi2 + i] = 0.0
            d_invdm[dsi2 + i] = 0.0

    def loop(self, d_idx, VIJ, XIJ, d_invdm, DWI, d_dvaux, s_m, s_idx, d_deaux,
             d_e, s_e):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        dim = self.dim
        dsi2 = d_idx * self.dimsq
        eij = d_e[d_idx] - s_e[s_idx]
        for row in range(dim):
            d_deaux[d_idx * dim + row] += s_m[s_idx] * eij * DWI[row]
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_dvaux[drowcol] += s_m[s_idx] * VIJ[row] * DWI[col]
                d_invdm[drowcol] += s_m[s_idx] * XIJ[row] * DWI[col]

    def post_loop(self, d_idx, d_invdm, d_dvaux, d_deaux):
        dsi2, row, col, rowcol, dim = declare('int', 5)
        invdm, idmat, dvaux, dvauxpre, dm = declare('matrix(9)', 5)
        auginvdm = declare('matrix(18)')
        deauxpre, deaux = declare('matrix(3)', 2)

        dim = self.dim
        dsi2 = self.dimsq * d_idx

        for row in range(dim):
            deauxpre[row] = d_deaux[dim * d_idx + row]
            for col in range(dim):
                rowcol = row * dim + col
                dvauxpre[rowcol] = d_dvaux[dsi2 + rowcol]
                invdm[rowcol] = d_invdm[dsi2 + rowcol]

        identity(idmat, dim)
        augmented_matrix(invdm, idmat, dim, dim, dim, auginvdm)
        gj_solve(auginvdm, dim, dim, dm)
        mat_mult(dm, dvauxpre, dim, dvaux)
        mat_vec_mult(dm, deauxpre, dim, deaux)

        for row in range(dim):
            d_deaux[d_idx * dim + row] = deaux[row]
            for col in range(dim):
                rowcol = row * dim + col
                d_dvaux[dsi2 + rowcol] = dvaux[rowcol]


class CorrectionMatrix(Equation):
    """
    Correction matrix, C, that accounts for the local particle distribution and
    used in calculation of gradients without using analytical derivatives of
    kernel.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [identity, augmented_matrix, gj_solve]

    def initialize(self, d_cm, d_idx):
        dsi, i, dimsq = declare('int', 3)
        dimsq = self.dimsq
        dsi = dimsq * d_idx
        for i in range(dimsq):
            d_cm[dsi + i] = 0.0

    def loop(self, d_idx, s_m, s_idx, XIJ, s_rho, d_cm, WI):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        mbbyrhob = s_m[s_idx] / s_rho[s_idx]
        for row in range(dim):
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_cm[drowcol] += mbbyrhob * XIJ[row] * XIJ[col] * WI

    def post_loop(self, d_idx, d_cm):
        invcm, cm, idmat = declare('matrix(9)', 3)
        augcm = declare('matrix(18)')
        dsi2, row, col, rowcol, dim, dimsq = declare('int', 6)

        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        identity(invcm, dim)
        identity(idmat, dim)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                invcm[rowcol] = d_cm[dsi2 + rowcol]

        augmented_matrix(invcm, idmat, dim, dim, dim, augcm)
        gj_solve(augcm, dim, dim, cm)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                d_cm[dsi2 + rowcol] = cm[rowcol]


class FirstGradient(Equation):
    """
    First gradient and divergence calculated using matrix inversion
    formulation without analytical derivative of the kernel.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult, mat_vec_mult]

    def initialize(self, d_dv, d_idx, d_divv, d_de):
        dsi2, i, dim, dimsq = declare('int', 4)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx

        for i in range(dim):
            d_de[dim * d_idx + i] = 0.0

        for i in range(dimsq):
            d_dv[dsi2 + i] = 0.0
        d_divv[d_idx] = 0.0

    def loop(self, d_idx, VIJ, XIJ, d_dv, WI, s_m, s_rho, s_idx, d_e, s_e,
             d_de):
        dsi2, row, col, dim = declare('int', 4)
        dim = self.dim
        dsi2 = d_idx * self.dimsq
        mbbyrhob = s_m[s_idx] / s_rho[s_idx]
        eij = d_e[d_idx] - s_e[s_idx]
        for row in range(dim):
            d_de[d_idx * dim + row] += mbbyrhob * eij * XIJ[row] * WI
            for col in range(dim):
                d_dv[dsi2 + row * dim + col] += mbbyrhob * VIJ[row] * \
                                                XIJ[col] * WI

    def post_loop(self, d_idx, d_dv, d_divv, d_cm, d_de):
        dv, dvpre, cm = declare('matrix(9)', 3)
        dsi2, row, col, rowcol, dim = declare('int', 5)
        depre, de = declare('matrix(3)', 2)
        dim = self.dim
        dsi2 = self.dimsq * d_idx

        for row in range(dim):
            depre[row] = d_de[dim * d_idx + row]
            for col in range(dim):
                rowcol = row * dim + col
                dvpre[rowcol] = d_dv[dsi2 + rowcol]
                cm[rowcol] = d_cm[dsi2 + rowcol]

        mat_mult(cm, dvpre, dim, dv)
        mat_vec_mult(cm, depre, dim, de)

        for row in range(dim):
            d_divv[d_idx] += dv[row * dim + row]
            d_de[d_idx * dim + row] = de[row]
            for col in range(dim):
                rowcol = row * dim + col
                d_dv[dsi2 + rowcol] = dv[rowcol]


class SecondGradient(Equation):
    """
    Second gradient calculated from auxiliary gradient using matrix inversion
    formulation without analytical derivative of the kernel.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        self.dimcu = self.dimsq * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult]

    def initialize(self, d_ddv, d_idx, d_dde):
        dsi3, i, dim, dimcu, blk, row, col = declare('int', 7)
        dsi2, dimsq = declare('int', 2)
        dim = self.dim
        dimsq = self.dimsq
        dimcu = self.dimcu
        dsi2 = dimsq * d_idx
        dsi3 = dimcu * d_idx
        for i in range(dimcu):
            d_ddv[dsi3 + i] = 0.0
        for i in range(dimsq):
            d_dde[dsi2 + i] = 0.0

    def loop(self, d_idx, XIJ, d_dvaux, s_dvaux, WI, d_ddv, s_m, s_rho,
             s_idx, s_deaux, d_deaux, d_dde):
        dsi2, row, col, dim, dimsq = declare('int', 5)
        blk, dblkrowcol, ssi2, srowcol, rowcol = declare('int', 5)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        ssi2 = s_idx * dimsq

        mbbyrhob = s_m[s_idx] / s_rho[s_idx]
        for row in range(dim):
            deij = d_deaux[d_idx * dim + row] - s_deaux[s_idx * dim + row]
            for col in range(dim):
                d_dde[dsi2 + row * dim + col] += mbbyrhob * deij * XIJ[col] * \
                                                 WI

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    dblkrowcol = dsi2 * dim + blk * dimsq + row * dim + col
                    dvij = d_dvaux[dsi2 + blk * dim + row] - s_dvaux[
                        ssi2 + blk * dim + row]
                    d_ddv[dblkrowcol] += mbbyrhob * dvij * XIJ[col] * WI

    def post_loop(self, d_idx, d_cm, d_ddv, d_dde):
        ddvpre = declare('matrix(27)')
        ddvpreb, ddvblk, cm, ddepre, dde = declare('matrix(9)', 5)
        dsi2, row, col, rowcol, dim, dimsq = declare('int', 6)
        blk, blkrowcol, dsi3 = declare('int', 3)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        dsi3 = dsi2 * dim

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                ddepre[rowcol] = d_dde[dsi2 + rowcol]
                cm[rowcol] = d_cm[dsi2 + rowcol]

        mat_mult(cm, ddepre, dim, dde)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                d_dde[dsi2 + rowcol] = dde[rowcol]

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    blkrowcol = blk * dimsq + row * dim + col
                    ddvpre[blkrowcol] = d_ddv[dsi3 + blkrowcol]

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    ddvpreb[rowcol] = ddvpre[blk * dimsq + rowcol]
            mat_mult(cm, ddvpreb, dim, ddvblk)
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    d_ddv[dsi3 + blk * dimsq + rowcol] = ddvblk[rowcol]


class EntropyBasedDissipationTrigger(Equation):
    """
    Simple, entropy-based dissipation trigger from [Rosswog2020a]_
    """

    def __init__(self, dest, sources, alphamax, alphamin, fkern, l0, l1,
                 gamma):
        self.alphamax = alphamax
        self.fkern = fkern
        self.l0 = l0
        self.l1 = l1
        self.gamma = gamma
        self.alphamin = alphamin
        super().__init__(dest, sources)

    def post_loop(self, d_h, d_idx, d_cs, d_alpha, d_s, d_p, d_rho, dt,
                  d_aalpha):
        snew = d_p[d_idx] / pow(d_rho[d_idx], self.gamma)
        tau = self.fkern * d_h[d_idx] / d_cs[d_idx]
        epsdot = abs(d_s[d_idx] - snew) * tau / (d_s[d_idx] * dt)
        d_s[d_idx] = snew
        ll = log(epsdot)
        x = min(max((ll - self.l0) / (self.l1 - self.l0), 0.0), 1.0)
        sx = ((6.0 * x - 15.0) * x + 10.0) * x * x * x
        alphades = self.alphamax * sx
        if d_alpha[d_idx] > alphades:
            d_aalpha[d_idx] = -(d_alpha[d_idx] - self.alphamin) / (30.0 * tau)
        else:
            d_alpha[d_idx] = alphades
            d_aalpha[d_idx] = 0.0


class WallBoundary(Equation):
    """:class:`WallBoundary
    <pysph.sph.gas_dynamics.boundary_equations.WallBoundary>` modified
    for MAGMA2.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        self.dimcu = self.dimsq * dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_h, d_htmp, d_h0,
                   d_u, d_v, d_w, d_wij, d_n, d_dndh, d_divv, d_alpha, d_ddv,
                   d_dv, d_de, d_cm, d_dde, d_rho0):
        i, dim, dimsq, dimcu, dsi1, dsi2, dsi3 = declare('int', 7)
        dim = self.dim
        dimsq = self.dimsq
        dimcu = self.dimcu
        dsi1 = d_idx * dim
        dsi2 = d_idx * dimsq
        dsi3 = d_idx * dimcu
        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_rho0[d_idx] = d_rho[d_idx]
        d_m[d_idx] = 0.0
        d_rho[d_idx] = 0.0
        d_e[d_idx] = 0.0
        d_cs[d_idx] = 0.0
        d_divv[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_h[d_idx] = d_h0[d_idx]
        d_htmp[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0
        d_alpha[d_idx] = 0.0

        for i in range(dim):
            d_de[dsi1 + i] = 0.0

        for i in range(dimsq):
            d_dv[dsi2 + i] = 0.0
            d_cm[dsi2 + i] = 0.0
            d_dde[dsi2 + i] = 0.0

        for i in range(dimcu):
            d_ddv[dsi3 + i] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_u, d_v,
             d_w, d_wij, d_htmp, s_p, s_rho, s_e, s_m, s_cs, s_h, s_divv, s_u,
             s_v, s_w, WI, s_n, d_n, s_dndh, d_dndh, d_alpha, s_alpha, d_de,
             s_de, d_dv, d_cm, d_dde, s_dv, s_cm, s_dde, s_ddv, d_ddv):

        i, dim, dimsq, dimcu = declare('int', 4)
        dsi1, dsi2, dsi3, ssi1, ssi2, ssi3 = declare('int', 6)
        dim = self.dim
        dimsq = self.dimsq
        dimcu = self.dimcu
        dsi1 = d_idx * dim
        dsi2 = d_idx * dimsq
        dsi3 = d_idx * dimcu
        ssi1 = s_idx * dim
        ssi2 = s_idx * dimsq
        ssi3 = s_idx * dimcu
        d_wij[d_idx] += WI
        d_p[d_idx] += s_p[s_idx] * WI
        d_u[d_idx] -= s_u[s_idx] * WI
        d_v[d_idx] -= s_v[s_idx] * WI
        d_w[d_idx] -= s_w[s_idx] * WI
        d_m[d_idx] += s_m[s_idx] * WI
        d_rho[d_idx] += s_rho[s_idx] * WI
        d_e[d_idx] += s_e[s_idx] * WI
        d_cs[d_idx] += s_cs[s_idx] * WI
        d_divv[d_idx] += s_divv[s_idx] * WI
        d_htmp[d_idx] += s_h[s_idx] * WI
        d_n[d_idx] += s_n[s_idx] * WI
        d_dndh[d_idx] += s_dndh[s_idx] * WI
        d_alpha[d_idx] += s_alpha[s_idx] * WI

        for i in range(dim):
            d_de[dsi1 + i] -= s_de[ssi1 + i] * WI

        for i in range(dimsq):
            d_dv[dsi2 + i] -= s_dv[ssi2 + i] * WI
            d_cm[dsi2 + i] += s_cm[ssi2 + i] * WI
            d_dde[dsi2 + i] += s_dde[ssi2 + i] * WI

        for i in range(dimcu):
            d_ddv[dsi3 + i] += s_ddv[ssi3 + i] * WI

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_h, d_u,
                  d_v, d_w, d_wij, d_htmp, d_n, d_dndh, d_de, d_dv, d_cm,
                  d_dde, d_ddv, d_rho0):
        i, dim, dimsq, dimcu, dsi1, dsi2, dsi3 = declare('int', 7)
        dim = self.dim
        dimsq = self.dimsq
        dimcu = self.dimcu
        dsi1 = d_idx * dim
        dsi2 = d_idx * dimsq
        dsi3 = d_idx * dimcu
        if d_wij[d_idx] > 1e-30:
            d_p[d_idx] = d_p[d_idx] / d_wij[d_idx]
            d_u[d_idx] = d_u[d_idx] / d_wij[d_idx]
            d_v[d_idx] = d_v[d_idx] / d_wij[d_idx]
            d_w[d_idx] = d_w[d_idx] / d_wij[d_idx]
            d_m[d_idx] = d_m[d_idx] / d_wij[d_idx]
            d_rho[d_idx] = d_rho[d_idx] / d_wij[d_idx]
            d_e[d_idx] = d_e[d_idx] / d_wij[d_idx]
            d_cs[d_idx] = d_cs[d_idx] / d_wij[d_idx]
            d_divv[d_idx] = d_divv[d_idx] / d_wij[d_idx]
            d_h[d_idx] = d_htmp[d_idx] / d_wij[d_idx]
            d_n[d_idx] = d_n[d_idx] / d_wij[d_idx]
            d_dndh[d_idx] = d_dndh[d_idx] / d_wij[d_idx]

            for i in range(dim):
                d_de[dsi1 + i] = d_de[dsi1 + i] / d_wij[d_idx]
            for i in range(dimsq):
                d_dv[dsi2 + i] = d_dv[dsi2 + i] / d_wij[d_idx]
                d_cm[dsi2 + i] = d_cm[dsi2 + i] / d_wij[d_idx]
                d_dde[dsi2 + i] = d_dde[dsi2 + i] / d_wij[d_idx]

            for i in range(dimcu):
                d_ddv[dsi3 + i] = d_ddv[dsi3 + i] / d_wij[d_idx]

        # Rho appears in denominator of correction matrix and elsewhere, so it
        # should not be zero.
        if abs(d_rho[d_idx]) < 1e-10:
            d_rho[d_idx] = d_rho0[d_idx]


class UpdateGhostProps(Equation):
    """
    :class:`MPMUpdateGhostProps
    <pysph.sph.gas_dynamics.basic.MPMUpdateGhostProps>` modified for MAGMA2.
    """

    def __init__(self, dest, dim, sources=None):
        super().__init__(dest, sources)
        self.dim = dim
        self.dimsq = dim * dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_tag, d_h, d_rho, d_dndh,
                   d_n, d_cm, d_dv, d_dvaux, d_ddv, d_dde, d_de, d_deaux,
                   d_cs, d_alpha):
        idx, dim, dimsq, row, col, rowcol, blk = declare('int', 7)
        blkrowcol, dsi2, si2, drowcol, srowcol = declare('int', 5)
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_h[d_idx] = d_h[idx]
            d_cs[d_idx] = d_cs[idx]
            d_alpha[d_idx] = d_alpha[idx]
            d_rho[d_idx] = d_rho[idx]
            d_dndh[d_idx] = d_dndh[idx]
            d_n[d_idx] = d_n[idx]
            dim = self.dim
            dimsq = self.dimsq
            dsi2 = dimsq * d_idx
            si2 = dimsq * idx
            for row in range(dim):
                d_de[d_idx * dim + row] = d_de[idx * dim + row]
                d_deaux[d_idx * dim + row] = d_de[idx * dim + row]
                for col in range(dim):
                    rowcol = row * dim + col
                    drowcol = dsi2 + rowcol
                    srowcol = si2 + rowcol
                    d_cm[drowcol] = d_cm[srowcol]
                    d_dv[drowcol] = d_dv[srowcol]
                    d_dvaux[drowcol] = d_dvaux[srowcol]
                    d_dde[drowcol] = d_dde[srowcol]

            for blk in range(dim):
                for row in range(dim):
                    for col in range(dim):
                        blkrowcol = blk * dimsq + row * dim + col
                        d_ddv[dim * dsi2 + blkrowcol] = d_ddv[dim * si2 +
                                                              blkrowcol]


class MomentumAndEnergy(Equation):
    def __init__(self, dest, sources, dim, fkern, eta_crit=0.3, eta_fold=0.2,
                 beta=2.0, alphac=0.05, eps=0.01):
        self.beta = beta
        self.dim = dim
        self.fkern = fkern
        self.dimsq = dim * dim
        self.eta_crit = eta_crit
        self.eta_fold = eta_fold
        self.alphac = alphac
        self.epssq = eps * eps
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [dot]

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0


class MomentumAndEnergyStdGrad(MomentumAndEnergy):
    """Standard Gradient formulation (stdGrad) momentum and energy equations
    with artificial viscosity and artificial conductivity from [Rosswog2020b]_
    """

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho, d_au,
             d_av, d_aw, d_ae, XIJ, VIJ, d_alpha, s_alpha, d_ddv, s_ddv,
             RHOIJ1, d_h, s_h, DWI, DWJ, d_dv, s_dv, d_de, s_de, d_dde, s_dde,
             d_e, s_e):
        etai, etaj, vij, mpinc, dvdel, ddvdeldel = declare('matrix(3)', 6)
        dsi2, ssi2, row, col, blk = declare('int', 5)
        blkrowcol, rowcol, dim, dimsq = declare('int', 4)

        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        ssi2 = dimsq * s_idx
        epssq = self.epssq
        beta = self.beta
        hi = self.fkern * d_h[d_idx]
        hj = self.fkern * s_h[s_idx]

        for row in range(dim):
            etai[row] = XIJ[row] / hi
            etaj[row] = XIJ[row] / hj

        # Limiter
        etaisq = dot(etai, etai, dim)
        etajsq = dot(etaj, etaj, dim)
        etaij = sqrt(min(etaisq, etajsq))

        aanum = 0.0
        aaden = 0.0
        for row in range(dim):
            mpinc[row] = 0.5 * XIJ[row]
            for col in range(dim):
                rowcol = row * dim + col
                aanum += d_dv[dsi2 + rowcol] * XIJ[row] * XIJ[col]
                aaden += s_dv[ssi2 + rowcol] * XIJ[row] * XIJ[col]

        aaij = aanum / aaden
        phiijin = min(1.0, 4.0 * aaij / ((1.0 + aaij) * (1.0 + aaij)))
        phiij = max(0.0, phiijin)

        if etaij < self.eta_crit:
            powin = (etaij - self.eta_crit) / self.eta_fold
            phiij = phiij * exp(-powin * powin)

        # Reconstruction
        dedel = 0.0
        ddedel = 0.0
        for row in range(dim):
            dvdel[row] = 0.0
            ddvdeldel[row] = 0.0

            # [(\partial_j e) \delta^j]_a - [(\partial_j e) \delta^j]_b
            dedel -= (d_de[d_idx * dim + row] +
                      s_de[s_idx * dim + row]) * mpinc[row]

            for col in range(dim):
                rowcol = row * dim + col

                # [(\partial_j v^i) \delta^j]_a - [(\partial_j v^i) \delta^j]_b
                dvdel[row] -= (d_dv[dsi2 + rowcol] +
                               s_dv[ssi2 + rowcol]) * mpinc[col]

                # [(\partial_l \partial_m e) \delta^l \delta^m]_a -
                # [(\partial_l \partial_m e) \delta^l \delta^m]_b
                ddedel += (d_dde[dsi2 + rowcol] -
                           s_dde[ssi2 + rowcol]) * mpinc[row] * mpinc[col]

                for blk in range(dim):
                    blkrowcol = dimsq * blk + row * dim + col

                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_a -
                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_b
                    ddvdeldel[row] += (d_ddv[dsi2 * dim + blkrowcol] - s_ddv[
                        ssi2 * dim + blkrowcol]) * mpinc[col] * mpinc[blk]

        # Reconstructed differences and norm(DWIJ)
        sm = 0.0
        for row in range(dim):
            vij[row] = VIJ[row] + phiij * (dvdel[row] + 0.5 * ddvdeldel[row])
            sm += (DWI[row] + DWJ[row]) * (DWI[row] + DWJ[row])
        eij = d_e[d_idx] - s_e[s_idx] + phiij * (dedel + 0.5 * ddedel)
        normdwij = 0.5 * sqrt(sm)

        # Artificial viscosity
        vsigng = sqrt(abs(d_p[d_idx] - s_p[s_idx]) * RHOIJ1)
        mui = min(0.0, dot(vij, etai, dim) / (etaisq + epssq))
        muj = min(0.0, dot(vij, etaj, dim) / (etajsq + epssq))
        qi = d_rho[d_idx] * mui * (-d_alpha[d_idx] * d_cs[d_idx] + beta * mui)
        qj = s_rho[s_idx] * muj * (-s_alpha[s_idx] * s_cs[s_idx] + beta * muj)
        pi = d_p[d_idx] + qi
        pj = s_p[s_idx] + qj

        # Accelerations for velocity
        mjpibyrhoisq = s_m[s_idx] * pi / (d_rho[d_idx] * d_rho[d_idx])
        mjpjbyrhojsq = s_m[s_idx] * pj / (s_rho[s_idx] * s_rho[s_idx])

        d_au[d_idx] -= mjpibyrhoisq * DWI[0] + mjpjbyrhojsq * DWJ[0]
        d_av[d_idx] -= mjpibyrhoisq * DWI[1] + mjpjbyrhojsq * DWJ[1]
        d_aw[d_idx] -= mjpibyrhoisq * DWI[2] + mjpjbyrhojsq * DWJ[2]

        # Accelerations for the thermal energy
        vijdotdwi = dot(VIJ, DWI, dim)
        d_ae[d_idx] += mjpibyrhoisq * vijdotdwi

        # artificial conduction
        d_ae[d_idx] -= (self.alphac * s_m[s_idx] * vsigng * eij * normdwij *
                        RHOIJ1)


class MomentumAndEnergyMI1(MomentumAndEnergy):
    """Matrix inversion formulation 1 (MI1) momentum and energy equations with
    artificial viscosity and artificial conductivity from [Rosswog2020b]_
    """

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho, d_au,
             d_av, d_aw, d_ae, XIJ, VIJ, d_alpha, s_alpha, d_ddv, s_ddv,
             RHOIJ1, d_h, s_h, d_cm, s_cm, WI, WJ, d_dv, s_dv, d_de, s_de,
             d_dde, s_dde, d_e, s_e):
        gmi, gmj, etai, etaj, vij, mpinc = declare('matrix(3)', 6)
        dvdel, ddvdeldel = declare('matrix(3)', 3)
        dsi2, ssi2, row, col, blk = declare('int', 5)
        blkrowcol, rowcol, dim, dimsq = declare('int', 4)

        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        ssi2 = dimsq * s_idx
        epssq = self.epssq
        beta = self.beta
        hi = self.fkern * d_h[d_idx]
        hj = self.fkern * s_h[s_idx]

        for row in range(3):
            gmi[row] = 0.0
            gmj[row] = 0.0

        for row in range(dim):
            etai[row] = XIJ[row] / hi
            etaj[row] = XIJ[row] / hj

        # Limiter
        etaisq = dot(etai, etai, dim)
        etajsq = dot(etaj, etaj, dim)
        etaij = sqrt(min(etaisq, etajsq))

        aanum = 0.0
        aaden = 0.0
        for row in range(dim):
            mpinc[row] = 0.5 * XIJ[row]
            for col in range(dim):
                rowcol = row * dim + col
                aanum += d_dv[dsi2 + rowcol] * XIJ[row] * XIJ[col]
                aaden += s_dv[ssi2 + rowcol] * XIJ[row] * XIJ[col]

        aaij = aanum / aaden
        phiijin = min(1.0, 4.0 * aaij / ((1.0 + aaij) * (1.0 + aaij)))
        phiij = max(0.0, phiijin)

        if etaij < self.eta_crit:
            powin = (etaij - self.eta_crit) / self.eta_fold
            phiij = phiij * exp(-powin * powin)

        # Reconstruction
        dedel = 0.0
        ddedel = 0.0
        for row in range(dim):
            dvdel[row] = 0.0
            ddvdeldel[row] = 0.0

            # [(\partial_j e) \delta^j]_a - [(\partial_j e) \delta^j]_b
            dedel -= (d_de[d_idx * dim + row] +
                      s_de[s_idx * dim + row]) * mpinc[row]

            for col in range(dim):
                rowcol = row * dim + col

                # [(\partial_j v^i) \delta^j]_a - [(\partial_j v^i) \delta^j]_b
                dvdel[row] -= (d_dv[dsi2 + rowcol] +
                               s_dv[ssi2 + rowcol]) * mpinc[col]

                # [(\partial_l \partial_m e) \delta^l \delta^m]_a -
                # [(\partial_l \partial_m e) \delta^l \delta^m]_b
                ddedel += (d_dde[dsi2 + rowcol] -
                           s_dde[ssi2 + rowcol]) * mpinc[row] * mpinc[col]

                for blk in range(dim):
                    blkrowcol = dimsq * blk + row * dim + col

                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_a -
                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_b
                    ddvdeldel[row] += (d_ddv[dsi2 * dim + blkrowcol] - s_ddv[
                        ssi2 * dim + blkrowcol]) * mpinc[col] * mpinc[blk]

        # Gradient functions and reconstructed differences
        sm = 0.0
        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                gmi[row] -= d_cm[dsi2 + rowcol] * XIJ[col] * WI
                gmj[row] -= s_cm[ssi2 + rowcol] * XIJ[col] * WJ
            gmij = 0.5 * (gmi[row] + gmj[row])
            sm += gmij * gmij
            vij[row] = VIJ[row] + phiij * (dvdel[row] + 0.5 * ddvdeldel[row])
        normgmij = 0.5 * sqrt(sm)
        eij = d_e[d_idx] - s_e[s_idx] + phiij * (dedel + 0.5 * ddedel)

        # Artificial viscosity
        vsigng = sqrt(abs(d_p[d_idx] - s_p[s_idx]) * RHOIJ1)
        mui = min(0.0, dot(vij, etai, dim) / (etaisq + epssq))
        muj = min(0.0, dot(vij, etaj, dim) / (etajsq + epssq))
        qi = d_rho[d_idx] * mui * (-d_alpha[d_idx] * d_cs[d_idx] + beta * mui)
        qj = s_rho[s_idx] * muj * (-s_alpha[s_idx] * s_cs[s_idx] + beta * muj)
        pi = d_p[d_idx] + qi
        pj = s_p[s_idx] + qj

        # Accelerations for velocity
        mjpibyrhoisq = s_m[s_idx] * pi / (d_rho[d_idx] * d_rho[d_idx])
        mjpjbyrhojsq = s_m[s_idx] * pj / (s_rho[s_idx] * s_rho[s_idx])

        d_au[d_idx] -= mjpibyrhoisq * gmi[0] + mjpjbyrhojsq * gmj[0]
        d_av[d_idx] -= mjpibyrhoisq * gmi[1] + mjpjbyrhojsq * gmj[1]
        d_aw[d_idx] -= mjpibyrhoisq * gmi[2] + mjpjbyrhojsq * gmj[2]

        # Accelerations for the thermal energy
        vijdotdwi = dot(VIJ, gmi, dim)
        d_ae[d_idx] += mjpibyrhoisq * vijdotdwi

        # artificial conduction
        d_ae[d_idx] -= (self.alphac * s_m[s_idx] * vsigng * eij * normgmij *
                        RHOIJ1)


class MomentumAndEnergyMI2(MomentumAndEnergy):
    """Matrix inversion formulation 2 (MI2) momentum and energy equations with
    artificial viscosity and artificial conductivity from [Rosswog2020b]_
    """

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho, d_au,
             d_av, d_aw, d_ae, XIJ, VIJ, d_alpha, s_alpha, d_ddv, s_ddv,
             RHOIJ1, d_h, s_h, d_cm, s_cm, WI, WJ, d_dv, s_dv, d_de, s_de,
             d_dde, s_dde, d_e, s_e):
        gmij, etai, etaj, vij, mpinc = declare('matrix(3)', 5)
        dvdel, ddvdeldel = declare('matrix(3)', 3)
        dsi2, ssi2, row, col, blk = declare('int', 5)
        blkrowcol, rowcol, dim, dimsq = declare('int', 4)

        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        ssi2 = dimsq * s_idx
        epssq = self.epssq
        beta = self.beta
        hi = self.fkern * d_h[d_idx]
        hj = self.fkern * s_h[s_idx]

        for row in range(dim):
            etai[row] = XIJ[row] / hi
            etaj[row] = XIJ[row] / hj

        for row in range(3):
            gmij[row] = 0.0

        # Limiter
        etaisq = dot(etai, etai, dim)
        etajsq = dot(etaj, etaj, dim)
        etaij = sqrt(min(etaisq, etajsq))

        aanum = 0.0
        aaden = 0.0
        for row in range(dim):
            mpinc[row] = 0.5 * XIJ[row]
            for col in range(dim):
                rowcol = row * dim + col
                aanum += d_dv[dsi2 + rowcol] * XIJ[row] * XIJ[col]
                aaden += s_dv[ssi2 + rowcol] * XIJ[row] * XIJ[col]

        aaij = aanum / aaden
        phiijin = min(1.0, 4.0 * aaij / ((1.0 + aaij) * (1.0 + aaij)))
        phiij = max(0.0, phiijin)

        if etaij < self.eta_crit:
            powin = (etaij - self.eta_crit) / self.eta_fold
            phiij = phiij * exp(-powin * powin)

        # Reconstruction
        dedel = 0.0
        ddedel = 0.0
        for row in range(dim):
            dvdel[row] = 0.0
            ddvdeldel[row] = 0.0

            # [(\partial_j e) \delta^j]_a - [(\partial_j e) \delta^j]_b
            dedel -= (d_de[d_idx * dim + row] +
                      s_de[s_idx * dim + row]) * mpinc[row]

            for col in range(dim):
                rowcol = row * dim + col

                # [(\partial_j v^i) \delta^j]_a - [(\partial_j v^i) \delta^j]_b
                dvdel[row] -= (d_dv[dsi2 + rowcol] +
                               s_dv[ssi2 + rowcol]) * mpinc[col]

                # [(\partial_l \partial_m e) \delta^l \delta^m]_a -
                # [(\partial_l \partial_m e) \delta^l \delta^m]_b
                ddedel += (d_dde[dsi2 + rowcol] -
                           s_dde[ssi2 + rowcol]) * mpinc[row] * mpinc[col]

                for blk in range(dim):
                    blkrowcol = dimsq * blk + row * dim + col

                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_a -
                    # [(\partial_l \partial_m v^i) \delta^l \delta^m]_b
                    ddvdeldel[row] += (d_ddv[dsi2 * dim + blkrowcol] - s_ddv[
                        ssi2 * dim + blkrowcol]) * mpinc[col] * mpinc[blk]

        # Gradient functions and reconstructed differences
        sm = 0.0
        for row in range(dim):
            gmi = 0.0
            gmj = 0.0
            for col in range(dim):
                rowcol = row * dim + col
                gmi -= d_cm[dsi2 + rowcol] * XIJ[col] * WI
                gmj -= s_cm[ssi2 + rowcol] * XIJ[col] * WJ
            gmij[row] = 0.5 * (gmi + gmj)
            sm += gmij[row] * gmij[row]
            vij[row] = VIJ[row] + phiij * (dvdel[row] + 0.5 * ddvdeldel[row])
        normgmij = sqrt(sm)
        eij = d_e[d_idx] - s_e[s_idx] + phiij * (dedel + 0.5 * ddedel)

        # Artificial viscosity
        vsigng = sqrt(abs(d_p[d_idx] - s_p[s_idx]) * RHOIJ1)
        mui = min(0.0, dot(vij, etai, dim) / (etaisq + epssq))
        muj = min(0.0, dot(vij, etaj, dim) / (etajsq + epssq))
        qi = d_rho[d_idx] * mui * (-d_alpha[d_idx] * d_cs[d_idx] + beta * mui)
        qj = s_rho[s_idx] * muj * (-s_alpha[s_idx] * s_cs[s_idx] + beta * muj)
        pi = d_p[d_idx] + qi
        pj = s_p[s_idx] + qj

        # Accelerations for velocity
        invrhosq = 1.0 / (d_rho[d_idx] * s_rho[s_idx])
        comn = s_m[s_idx] * (pi + pj) * invrhosq

        d_au[d_idx] -= comn * gmij[0]
        d_av[d_idx] -= comn * gmij[1]
        d_aw[d_idx] -= comn * gmij[2]

        # Accelerations for the thermal energy
        vijdotgmij = dot(VIJ, gmij, dim)
        d_ae[d_idx] -= self.alphac * s_m[
            s_idx] * vsigng * eij * normgmij * RHOIJ1
        d_ae[d_idx] += s_m[s_idx] * pi * invrhosq * vijdotgmij


class EvaluateTildeMu(Equation):
    """
    Find :math:`\\tilde{\\mu}` to calculate time step.
    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [dot]

    def initialize(self, d_idx, d_tilmu):
        d_tilmu[d_idx] = -INFINITY

    def loop(self, d_tilmu, d_idx, d_h, VIJ, XIJ, R2IJ):
        d_tilmu[d_idx] = max(d_tilmu[d_idx],
                             d_h[d_idx] * dot(VIJ, XIJ, self.dim) / (
                                     R2IJ + 0.01))


class SettleByArtificialPressure(Equation):
    """Equation 40 of [Rosswog2020b]_ .
    Combined with an equation to update density (and smoothing length, if
    preferred), this equation can be evaluated through
    :py:class:`~pysph.tools.sph_evaluator.SPHEvaluator` to settle the
    particles and obtain an initial distribution."""

    def __init__(self, dest, sources, xi=0.5, fkern=1.0):
        self.fkern = fkern
        self.xi = xi
        super().__init__(dest, sources)

    def initialize(self, d_deltax, d_deltay, d_deltaz, d_idx, d_n,
                   d_pouerr):
        d_deltax[d_idx] = 0.0
        d_deltay[d_idx] = 0.0
        d_deltaz[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_pouerr[d_idx] = 0.0  # partition of unity error

    def loop(self, d_rho, d_idx, d_rhodes, s_rho, s_rhodes, s_idx,
             d_deltax, d_deltay, d_deltaz, DWI, d_n, WI, s_m, d_pouerr):
        cpia = max(1 + ((d_rho[d_idx] - d_rhodes[d_idx]) / d_rhodes[d_idx]),
                   0.1)
        cpib = max(1 + ((s_rho[s_idx] - s_rhodes[s_idx]) / s_rhodes[s_idx]),
                   0.1)

        common = (cpia + cpib) / s_rho[s_idx]

        d_deltax[d_idx] += common * DWI[0]
        d_deltay[d_idx] += common * DWI[1]
        d_deltaz[d_idx] += common * DWI[2]

        d_n[d_idx] += WI
        d_pouerr[d_idx] += s_m[d_idx] * WI / s_rho[s_idx]

    def post_loop(self, d_deltax, d_deltay, d_deltaz, d_idx, d_h, d_m,
                  d_pouerr, d_rhodes, d_n, d_x, d_y, d_z):
        xi = self.xi
        hi = self.fkern * d_h[d_idx]
        common = -xi * hi * hi * d_m[d_idx]
        d_deltax[d_idx] *= common
        d_deltay[d_idx] *= common
        d_deltaz[d_idx] *= common

        d_x[d_idx] += d_deltax[d_idx]
        d_y[d_idx] += d_deltay[d_idx]
        d_z[d_idx] += d_deltaz[d_idx]

        d_pouerr[d_idx] = 1 - d_pouerr[d_idx]
        d_m[d_idx] = d_rhodes[d_idx] / d_n[d_idx]


class TVDRK2Step(IntegratorStep):
    """
    Total variation diminishing (TVD) second-order Runge–Kutta (RK2)
    integrator step.
    """

    def initialize(self, d_idx, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_converged,
                   d_au0, d_av0, d_aw0, d_ae0, d_ah0, d_arho0, d_an0,
                   d_aalpha0, d_au, d_av, d_aw, d_ae, d_ah, d_arho, d_an,
                   d_aalpha):
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_au0[d_idx] = d_au[d_idx]
        d_av0[d_idx] = d_av[d_idx]
        d_aw0[d_idx] = d_aw[d_idx]

        d_ae0[d_idx] = d_ae[d_idx]

        d_ah0[d_idx] = d_ah[d_idx]
        d_arho0[d_idx] = d_arho[d_idx]
        d_an0[d_idx] = d_an[d_idx]
        d_aalpha0[d_idx] = d_aalpha[d_idx]

        # set the converged attribute to 0 at the beginning of a Group
        d_converged[d_idx] = 0

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_e, d_au, d_av,
               d_aw, d_ae, d_rho, d_arho, d_h, d_ah, dt, d_n, d_an, d_alpha,
               d_aalpha, d_h0, d_converged):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        # update thermal energy
        d_e[d_idx] += dt * d_ae[d_idx]

        # predict density and smoothing lengths for faster
        # convergence. NNPS need not be explicitly updated since it
        # will be called at the end of the predictor stage.
        d_h0[d_idx] = d_h[d_idx]
        d_h[d_idx] += dt * d_ah[d_idx]
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_n[d_idx] += dt * d_an[d_idx]
        d_alpha[d_idx] += dt * d_aalpha[d_idx]
        d_converged[d_idx] = 0

    def stage2(self, d_idx, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w,
               d_e, d_au, d_av, d_aw, d_ae, dt, d_alpha, d_aalpha, d_h,
               d_tilmu, d_cs, d_dt_adapt, d_au0, d_av0, d_aw0, d_aalpha0,
               d_ae0, d_h0, d_ah0, d_ah, d_arho, d_arho0, d_an, d_an0,
               d_rho, d_n):
        dtb2 = 0.5 * dt

        d_x[d_idx] += dtb2 * (d_u[d_idx] - d_u0[d_idx])
        d_y[d_idx] += dtb2 * (d_v[d_idx] - d_v0[d_idx])
        d_z[d_idx] += dtb2 * (d_w[d_idx] - d_w0[d_idx])

        d_u[d_idx] += dtb2 * (d_au[d_idx] - d_au0[d_idx])
        d_v[d_idx] += dtb2 * (d_av[d_idx] - d_av0[d_idx])
        d_w[d_idx] += dtb2 * (d_aw[d_idx] - d_aw0[d_idx])

        d_e[d_idx] += dtb2 * (d_ae[d_idx] - d_ae0[d_idx])
        d_alpha[d_idx] += dtb2 * (d_aalpha[d_idx] - d_aalpha0[d_idx])

        d_h0[d_idx] = d_h[d_idx]
        d_h[d_idx] += dtb2 * (d_ah[d_idx] - d_ah0[d_idx])
        d_rho[d_idx] += dtb2 * (d_arho[d_idx] - d_arho0[d_idx])
        d_n[d_idx] += dtb2 * (d_an[d_idx] - d_an0[d_idx])
        d_alpha[d_idx] += dtb2 * (d_aalpha[d_idx] - d_aalpha0[d_idx])

        # For adaptive time-stepping
        fmag = sqrt(d_au[d_idx] * d_au[d_idx] + d_av[d_idx] * d_av[d_idx] +
                    d_aw[d_idx] * d_aw[d_idx])

        dt_force = sqrt(d_h[d_idx] / fmag)
        dt_courant_visc = d_h[d_idx] / (d_cs[d_idx] + 0.6 * d_alpha[d_idx] * (
                d_cs[d_idx] + 2.0 * d_tilmu[d_idx]))

        d_dt_adapt[d_idx] = 0.2 * min(dt_force, dt_courant_visc)


class TVDRK2Integrator(Integrator):
    r"""
    Total variation diminishing (TVD) second-order Runge–Kutta (RK2)
    integrator. Prescribed equations in [Rosswog2020b]_ are,

    .. math::

        y^{*} = y^n + \Delta t f(y^{n}) --> Predict

        y^{n+1} = 0.5 (y^n + y^{*} + \Delta t f(y^{*})) --> Correct

    This is not suitable to be used with periodic boundaries. Say, if
    a particle crosses the left boundary at the prediction step,
    `update_domain()` will introduce that particle at the right boundary.
    Afterwards, the correction step essentially averages the positions and the
    particle ends up near the mid-point. To do away with this issue, the
    equation for the correction step is changed to,

    .. math::

        y^{n+1} = y^{*} + 0.5 * \Delta t (f(y^{*}) - f(y^{n}))

    """

    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations()
        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5 * dt, 1)
        self.compute_accelerations()

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


class TVDRK2IntegratorWithRecycling(Integrator):
    r"""
    Total variation diminishing (TVD) second-order Runge–Kutta (RK2)
    integrator with recycling of derivatives. The system is advanced using:

    .. math::

        y^{*,n} = y^n + \Delta t f(y^{*,n-1})

        y^{n+1} = 0.5 (y^n + y^{*} + \Delta t f(y^{*,n}))

    This is not suitable to be used with periodic boundaries. Say, if
    a particle crosses the left boundary at the prediction step,
    `update_domain()` will introduce that particle at the right boundary.
    Afterwards, the correction step essentially averages the positions and the
    particle ends up near the mid-point. To do away with this issue, the
    equation for correction step is changed to,

    .. math::

        y^{n+1} = y^{*} + 0.5 * \Delta t (f(y^{*,n}) - f(y^{*,n-1}))
    """

    def one_timestep(self, t, dt):
        self.initialize()

        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5 * dt, 1)
        self.compute_accelerations()

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


@annotate(fst='int', lst='int', arr='doublep')
def quicksort(arr, fst=0, lst=3):
    """
    Sort in-place with QuickSort
    Thanks https://stackoverflow.com/a/31102672

    Parameters
    ----------
    arr : list
        Array to be sorted.
    fst : int
        The index of the first element from arr and key to begin sorting from.
        Must be in the range [0, len(xs)).
    lst : int
        The index of the last element from arr and key to begin sorting from.
        Must be in the range [0, len(xs)).

    """
    i, j = declare('int', 2)
    if fst >= lst:
        return

    i, j = fst, lst
    pivot = arr[lst]

    while i <= j:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1

        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
            i, j = i + 1, j - 1
    quicksort(arr, fst, j)
    quicksort(arr, i, lst)
