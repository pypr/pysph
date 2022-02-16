"""
References
-----------

    .. [CullenDehnen2010] Cullen, Lee, and Walter Dehnen. “Inviscid Smoothed
        Particle Hydrodynamics: Inviscid Smoothed Particle Hydrodynamics.”
        Monthly Notices of the Royal Astronomical Society 408, no. 2
        (October 21, 2010): 669–83.
        https://doi.org/10.1111/j.1365-2966.2010.17158.x.

    .. [ReadHayfield2012] Read, J. I., and T. Hayfield. “SPHS: Smoothed
        Particle Hydrodynamics with a Higher Order Dissipation Switch:
        SPH with a Higher Order Dissipation Switch.” Monthly Notices of the
        Royal Astronomical Society 422, no. 4 (June 1, 2012): 3037–55.
        https://doi.org/10.1111/j.1365-2966.2012.20819.x.
    """
from compyle.types import declare
from pysph.base.particle_array import get_ghost_tag

from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.wc.linalg import augmented_matrix, gj_solve, identity, mat_mult

GHOST_TAG = get_ghost_tag()


class PSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, hfact, betab=2.0, fkern=1.0,
                 max_density_iterations=250, alphac=0.25,
                 density_iteration_tolerance=1e-3, has_ghosts=False,
                 alphamin=0.02, alphamax=2.0, betac=0.7, betad=0.05,
                 betaxi=1.0):
        """
        Pressure-energy formulation [Hopkins2013]_ including Cullen-Dehnen
        artificial viscocity switch [CullenDehnen2010]_ with modifications,
        as presented in Appendix F2 of [Hopkins2015]_ .

        Notes
        -----
        Is this exactly in accordance with what is proposed in [Hopkins2015]_ ?
            Not quite.

        What is different then?
            #. Adapting smoothing length using MPM [KP14]_ procedure from
               :class:`SummationDensity
               <pysph.sph.gas_dynamics.basic.SummationDensity>`. In this,
               calculation of grad-h terms are changed to that specified for
               this scheme.
            #. Using the PEC integrator step. No individual
               adaptive time-stepping.
            #. Using :class:`Gaussian Kernel <pysph.base.kernels.Gaussian>`
               by default instead of Cubic Spline with radius scale 1.

        Tip: Reduce the number of points if particle penetration is
        encountered. This has to be done while running
        ``gas_dynamics.wc_blastwave`` and ``gas_dynamics.robert``.

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
            as kernel_factor in other gas dynamics schemes.
        betab : float, optional
            :math:`\\beta_b` for artificial viscosity, by default 2.0
        fkern : float, optional
            :math:`f_{kern}`, Factor to scale smoothing length for equivalence
            with classic kernel when using kernel with altered
            `radius_scale` is being used, by default 1.
        max_density_iterations : int, optional
            Maximum number of iterations to run for one density step,
            by default 250.
        alphac : float, optional
            :math:`\\alpha_c` for artificial conductivity, by default 0.25
        density_iteration_tolerance : float, optional
            Maximum difference allowed in two successive density iterations,
            by default 1e-3
        has_ghosts : bool, optional
            if ghost particles (either mirror or periodic) is used, by default
            False
        alphamin : float, optional
            :math:`\\alpha_{min}` for artificial viscosity switch, by default
            0.02
        alphamax : float, optional
            :math:`\\alpha_{max}` for artificial viscosity switch, by default
            2.0
        betac : float, optional
            :math:`\\beta_c` for artificial viscosity switch, by default 0.7
        betad : float, optional
            :math:`\\beta_d` for artificial viscosity switch, by default 0.05
        betaxi : float, optional
            :math:`\\beta_{\\xi}` for artificial viscosity switch,
            by default 1.0
        """

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.betab = betab
        self.hfact = hfact
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts
        self.fkern = fkern
        self.alphac = alphac
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.betac = betac
        self.betad = betad
        self.betaxi = betaxi

    def add_user_options(self, group):
        group.add_argument("--alpha-max", action="store", type=float,
                           dest="alphamax", default=None,
                           help="alpha_max for artificial viscosity switch. ")

        group.add_argument("--alpha-min", action="store", type=float,
                           dest="alphamin", default=None,
                           help="alpha_min for artificial viscosity switch. ")

        group.add_argument("--beta-b", action="store", type=float,
                           dest="betab", default=None,
                           help="beta for the artificial viscosity.")

        group.add_argument("--beta-xi", action="store", type=float,
                           dest="betaxi", default=None,
                           help="beta_xi for artificial viscosity switch.")

        group.add_argument("--beta-d", action="store", type=float,
                           dest="betad", default=None,
                           help="beta_d for artificial viscosity switch.")

        group.add_argument("--beta-c", action="store", type=float,
                           dest="betac", default=None,
                           help="beta_c for artificial viscosity switch.")

        group.add_argument("--alpha-c", action="store", type=float,
                           dest="alphac", default=None,
                           help="alpha_c for artificial conductivity. ")

        group.add_argument("--gamma", action="store", type=float, dest="gamma",
                           default=None, help="gamma for the state equation.")

    def consume_user_options(self, options):
        vars = ['gamma', 'alphamax', 'alphamin', 'alphac', 'betab', 'betaxi',
                'betad', 'betac']
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):

        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        if hasattr(kernel, 'fkern'):
            self.fkern = kernel.fkern
        else:
            self.fkern = 1.0

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = PECStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group

        equations = []
        # Find the optimal 'h'
        all_pa = self.fluids + self.solids

        g1 = []
        for fluid in self.fluids:
            g1.append(PSPHSummationDensityAndPressure(
                dest=fluid, sources=all_pa, hfact=self.hfact,
                density_iterations=True, dim=self.dim,
                htol=self.density_iteration_tolerance, gamma=self.gamma))
            equations.append(
                Group(equations=g1, update_nnps=True, iterate=True,
                      max_iterations=self.max_density_iterations))

        g2 = []
        for fluid in self.fluids:
            g2.append(GradientKinsfolkC1(
                dest=fluid, sources=all_pa, dim=self.dim))
            g2.append(SignalVelocity(dest=fluid, sources=all_pa))
        equations.append(Group(equations=g2))

        g3 = []
        for fluid in self.fluids:
            g3.append(LimiterAndAlphas(dest=fluid, sources=all_pa,
                                       alphamin=self.alphamin,
                                       alphamax=self.alphamax,
                                       betac=self.betac, betad=self.betad,
                                       betaxi=self.betaxi, fkern=self.fkern))
        equations.append(Group(equations=g3))

        g4 = []
        for solid in self.solids:
            g4.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g4))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(UpdateGhostProps(dest=fluid, sources=None))
            equations.append(Group(equations=gh, real=False))

        g5 = []
        for fluid in self.fluids:
            g5.append(MomentumAndEnergy(
                dest=fluid, sources=all_pa, dim=self.dim, betab=self.betab,
                fkern=self.fkern, alphac=self.alphac, gamma=self.gamma))
        equations.append(Group(equations=g5))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])

        props = ['rho', 'm', 'x', 'y', 'z', 'u', 'v', 'w', 'h', 'cs', 'p', 'e',
                 'au', 'av', 'aw', 'ae', 'pid', 'gid', 'tag', 'dwdh', 'h0',
                 'converged', 'ah', 'arho', 'e0', 'rho0', 'u0', 'v0', 'w0',
                 'x0', 'y0', 'z0']
        more_props = ['drhosumdh', 'n', 'dndh', 'prevn', 'prevdndh',
                      'prevdrhosumdh', 'divv', 'dpsumdh', 'dprevpsumdh', 'an',
                      'adivv', 'trssdsst', 'vsig', 'alpha', 'alpha0', 'xi']
        props.extend(more_props)
        output_props = 'rho p u v w x y z e n divv h alpha'.split(' ')
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            # Guess for number density.
            pa.add_property('n', data=pa.rho / pa.m)
            pa.add_property('gradv', stride=9)
            pa.add_property('invtt', stride=9)
            pa.add_property('grada', stride=9)
            pa.add_property('ss', stride=6)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('m0 wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)


class PSPHSummationDensityAndPressure(Equation):
    def __init__(self, dest, sources, dim, gamma, density_iterations=False,
                 iterate_only_once=False, hfact=1.2, htol=1e-6):
        """
        :class:`SummationDensity
        <pysph.sph.gas_dynamics.basic.SummationDensity>` modified to use
        number density for calculation of grad-h terms and to calculate
        pressure and speed of sound as well.

        Ref. Appendix F2 [Hopkins2015]_

        Parameters
        ----------
        density_iterations : bint, optional
            Flag to indicate density iterations are required, by default False
        iterate_only_once : bint, optional
            Flag to indicate if only one iteration is required,
             by default False
        hfact : float, optional
            :math:`h_{fact}`, by default 1.2
        htol : double, optional
            Iteration tolerance, by default 1e-6
        """

        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.hfact = hfact
        self.htol = htol
        self.equation_has_converged = 1
        self.gamma = gamma
        self.gammam1 = gamma - 1.0

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_n, d_dndh, d_prevn,
                   d_prevdndh, d_p, d_dpsumdh, d_dprevpsumdh, d_an):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0
        d_an[d_idx] = 0.0

        d_p[d_idx] = 0.0
        d_dprevpsumdh[d_idx] = d_dpsumdh[d_idx]
        d_dpsumdh[d_idx] = 0.0

        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, s_m, VIJ, WI, DWI, GHI, d_n,
             d_dndh, d_h, d_prevn, d_prevdndh, s_e, d_p, d_dpsumdh, d_e, d_an):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        mj_wi = mj * WI
        d_rho[d_idx] += mj_wi
        d_p[d_idx] += self.gammam1 * s_e[s_idx] * mj_wi

        # number density accelerations
        hibynidim = d_h[d_idx] / (d_prevn[d_idx] * self.dim)
        inbrkti = 1 + d_prevdndh[d_idx] * hibynidim
        inprthsi = d_dpsumdh[d_idx] * hibynidim / (
                self.gammam1 * s_m[s_idx] * d_e[d_idx])
        fij = 1 - inprthsi / inbrkti
        vijdotdwij_fij = vijdotdwij * fij
        d_an[d_idx] += vijdotdwij_fij

        # density acceleration is not essential as such
        d_arho[d_idx] += mj * vijdotdwij_fij

        # gradient of kernel w.r.t h
        d_dpsumdh[d_idx] += mj * self.gammam1 * d_e[d_idx] * GHI
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_rho, d_h0, d_h, d_ah, d_converged, d_cs, d_p,
                  d_n, d_dndh, d_an):

        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])

        # iteratively find smoothing length consistent with the
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


class GradientKinsfolkC1(Equation):
    def __init__(self, dest, sources, dim):
        """
        First order consistent,

            - Velocity gradient, grad(v)
            - Acceleration gradient, grad(a)
            - Velocity divergence, div(v)
            - Velocity divergence rate, d (div(v)) / dt
            - Traceless symmetric strain rate, S
            - trace(dot(S,transpose(S)))

        Ref. Appendix B [CullenDehnen2010]_
        """
        self.dim = dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, mat_mult]

    def initialize(self, d_gradv, d_idx, d_invtt, d_divv, d_grada, d_adivv,
                   d_trssdsst):
        start_indx, i, dim = declare('int', 3)
        start_indx = 9 * d_idx
        for i in range(9):
            d_gradv[start_indx + i] = 0.0
            d_invtt[start_indx + i] = 0.0
            d_grada[start_indx + i] = 0.0

        d_divv[d_idx] = 0.0
        d_adivv[d_idx] = 0.0
        d_trssdsst[d_idx] = 0.0

    def loop(self, d_idx, d_invtt, s_m, s_idx, VIJ, DWI, XIJ, d_gradv, d_grada,
             d_au, s_au, d_av, s_av, d_aw, s_aw):
        start_indx, row, col, drowcol, dim = declare('int', 5)
        dim = self.dim
        start_indx = d_idx * 9

        aij = declare('matrix(3)')

        aij[0] = d_au[d_idx] - s_au[s_idx]
        aij[1] = d_av[d_idx] - s_av[s_idx]
        aij[2] = d_aw[d_idx] - s_aw[s_idx]

        for row in range(dim):
            for col in range(dim):
                drowcol = start_indx + row * 3 + col
                d_invtt[drowcol] -= s_m[s_idx] * XIJ[row] * DWI[col]
                d_gradv[drowcol] -= s_m[s_idx] * VIJ[row] * DWI[col]
                d_grada[drowcol] -= s_m[s_idx] * aij[row] * DWI[col]

    def post_loop(self, d_idx, d_gradv, d_invtt, d_divv, d_grada, d_adivv,
                  d_ss, d_trssdsst):
        tt = declare('matrix(9)')
        invtt = declare('matrix(9)')
        augtt = declare('matrix(18)')
        idmat = declare('matrix(9)')
        gradv = declare('matrix(9)')
        grada = declare('matrix(9)')

        start_indx, row, col, rowcol, drowcol, dim, colrow = declare('int', 7)
        ltstart_indx, dltrowcol = declare('int', 2)
        dim = self.dim
        start_indx = 9 * d_idx
        identity(idmat, 3)
        identity(tt, 3)

        for row in range(3):
            for col in range(3):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                gradv[rowcol] = d_gradv[drowcol]
                grada[rowcol] = d_grada[drowcol]

        for row in range(dim):
            for col in range(dim):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                tt[rowcol] = d_invtt[drowcol]

        augmented_matrix(tt, idmat, 3, 3, 3, augtt)
        gj_solve(augtt, 3, 3, invtt)
        gradvls = declare('matrix(9)')
        gradals = declare('matrix(9)')
        mat_mult(gradv, invtt, 3, gradvls)
        mat_mult(grada, invtt, 3, gradals)

        for row in range(dim):
            d_divv[d_idx] += gradvls[row * 3 + row]
            d_adivv[d_idx] += gradals[row * 3 + row]
            for col in range(dim):
                rowcol = row * 3 + col
                colrow = row + col * 3
                drowcol = start_indx + rowcol
                d_gradv[drowcol] = gradvls[rowcol]
                d_grada[drowcol] = gradals[rowcol]
                d_adivv[d_idx] -= gradals[rowcol] * gradals[colrow]

        # Traceless Symmetric Strain Rate
        divvbydim = d_divv[d_idx] / dim
        start_indx = d_idx * 9
        ltstart_indx = d_idx * 6
        for row in range(dim):
            col = row
            rowcol = start_indx + row * 3 + col
            dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
            d_ss[dltrowcol] = d_gradv[rowcol] - divvbydim

        for row in range(1, dim):
            for col in range(row):
                rowcol = row * 3 + col
                colrow = row + col * 3
                dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
                d_ss[dltrowcol] = 0.5 * (gradvls[rowcol] + gradvls[colrow])

        # Trace ( S dot transpose(S) )
        for row in range(dim):
            for col in range(dim):
                dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
                d_trssdsst[d_idx] += d_ss[dltrowcol] * d_ss[dltrowcol]


class SignalVelocity(Equation):
    """
    Ref. Equation 25 [Hopkins2015]_
    """

    def initialize(self, d_idx, d_vsig):
        d_vsig[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_u, d_v, d_w, s_u,
                 s_v, s_w, d_cs, s_cs, d_vsig, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        vijdotxij = 0.0
        cij = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]

            vij[0] = d_u[d_idx] - s_u[s_idx]
            vij[1] = d_v[d_idx] - s_v[s_idx]
            vij[2] = d_w[d_idx] - s_w[s_idx]

            vijdotxij = vij[0] * xij[0] + vij[1] * xij[1] + vij[2] * xij[2]
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            d_vsig[d_idx] = max(d_vsig[d_idx], cij - min(0, vijdotxij))


class LimiterAndAlphas(Equation):
    def __init__(self, dest, sources, alphamin=0.02, alphamax=2.0, betac=0.7,
                 betad=0.05, betaxi=1.0, fkern=1.0):
        """
        Cullen Dehnen's limiter for artificial viscosity modified by Hopkins.

        Ref. Appendix F2 [Hopkins2015]_
        """
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.betac = betac
        self.betad = betad
        self.betaxi = betaxi
        self.fkern = fkern
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_xi):
        d_xi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_xi, s_divv, WI):

        if s_divv[s_idx] < 0:
            sign = -1.0
        else:
            sign = 1.0

        d_xi[d_idx] += sign * s_m[s_idx] * WI

    def post_loop(self, d_idx, d_xi, d_rho, d_h, d_adivv, d_cs, d_alpha0,
                  d_vsig, dt, d_divv, d_trssdsst, d_alpha):
        d_xi[d_idx] = 1.0 - d_xi[d_idx] / d_rho[d_idx]
        fhi = self.fkern * d_h[d_idx]

        if d_adivv[d_idx] >= 0 or d_divv[d_idx] >= 0:
            alphatmp = 0.0
        else:
            absadivv = abs(d_adivv[d_idx])
            csbyfhi = d_cs[d_idx] / fhi
            alphatmp = self.alphamax * absadivv / (
                    absadivv + self.betac * csbyfhi * csbyfhi)

        if alphatmp >= d_alpha0[d_idx]:
            d_alpha0[d_idx] = alphatmp
        elif alphatmp < d_alpha0[d_idx]:  # Isn't this obvious?
            epow = exp(-self.betad * dt * abs(d_vsig[d_idx]) * 0.5 / fhi)
            d_alpha0[d_idx] = alphatmp + (d_alpha0[d_idx] - alphatmp) * epow

        xip4 = d_xi[d_idx] * d_xi[d_idx] * d_xi[d_idx] * d_xi[d_idx]
        alnumtt = self.betaxi * xip4 * d_divv[d_idx]
        alnumt = alnumtt * alnumtt
        alnum = alnumt * d_alpha0[d_idx]
        alden = alnumt + d_trssdsst[d_idx]

        if alden < 1e-8:
            d_alpha[d_idx] = self.alphamin
        else:
            d_alpha[d_idx] = max(alnum / alden, self.alphamin)


class MomentumAndEnergy(Equation):
    def __init__(self, dest, sources, dim, fkern, gamma, betab=2.0,
                 alphac=0.25):
        r"""
        PSPH Momentum and Energy Equations with artificial viscosity and
        artificial conductivity.

        Possible typos in that have been taken care of,

            1. Instead of Equation F15 [Hopkins2015]_ for evolution of total
               energy sans artificial viscosity and artificial conductivity,

                .. math::
                    \frac{\mathrm{d} E_{i}}{\mathrm{~d} t}= \boldsymbol{v}_{i}
                    \cdot \frac{\mathrm{d} \boldsymbol{P}_{i}}{\mathrm{~d} t}-
                    \sum_{j=1}^{N}(\gamma-1)^{2} m_{i} m_{j} u_{i} u_{j}
                    \frac{f_{i j}}{\bar{P}_{i}}\left(\boldsymbol{v}_{i}-
                    \boldsymbol{v}_{j}\right) \cdot \nabla_{i} W_{i j}
                    \left(h_{i}\right),

               it should have been,

                .. math::
                    \frac{\mathrm{d} E_{i}}{\mathrm{~d} t}= \boldsymbol{v}_{i}
                    \cdot \frac{\mathrm{d} \boldsymbol{P}_{i}}{\mathrm{~d} t}+
                    \sum_{j=1}^{N}(\gamma-1)^{2} m_{i} m_{j} u_{i} u_{j}
                    \frac{f_{i j}}{\bar{P}_{i}}\left(\boldsymbol{v}_{i}-
                    \boldsymbol{v}_{j}\right) \cdot \nabla_{i} W_{i j}
                    \left(h_{i}\right).

               Specific thermal energy, :math:`u`, would therefore be evolved
               using,

                .. math::
                    \frac{\mathrm{d} u_{i}}{\mathrm{~d} t}=
                    \sum_{j=1}^{N}(\gamma-1)^{2} m_{j} u_{i} u_{j}
                    \frac{f_{i j}}{\bar{P}_{i}}\left(\boldsymbol{v}_{i}-
                    \boldsymbol{v}_{j}\right) \cdot \nabla_{i} W_{i j}
                    \left(h_{i}\right).

            #. Equation F18 [Hopkins2015]_ for contribution of
               artificial viscosity to the evolution of total
               energy is,

                .. math::
                    \frac{\mathrm{d} E_{i}}{\mathrm{~d} t}= \alpha_{\mathrm{C}}
                    \sum_{j} m_{i} m_{j} \alpha_{i j} \tilde{v}_{s}\left(u_{i}-
                    u_{j}\right) \times \frac{\left|P_{i}-P_{j}\right|}{P_{i}+
                    P_{j}} \frac{\nabla_{i} W_{i j}\left(h_{i}\right)+
                    \nabla_{i} W_{i j}\left(h_{j}\right)}{\bar{\rho}_{i}+
                    \bar{\rho}_{j}} .

               Carefully comparing with [ReadHayfield2012]_ and [KP14]_,
               specific thermal energy, :math:`u`, should be evolved
               using,

                .. math::
                    \frac{\mathrm{d} u_{i}}{\mathrm{~d} t}= \alpha_{\mathrm{C}}
                    \sum_{j} & m_{j} \alpha_{i j} \tilde{v}_{s}\left(u_{i}-
                    u_{j}\right) \frac{\left|P_{i}-P_{j}\right|}{P_{i}+
                    P_{j}} \\ & \frac{\nabla_{i} W_{i j}\left(h_{i}\right)+
                    \nabla_{i} W_{i j}\left(h_{j}\right)}{\bar{\rho}_{i}+
                    \bar{\rho}_{j}} \cdot \frac{\left(\boldsymbol{x}_{i}-
                    \boldsymbol{x}_{j}\right)}{\left|\boldsymbol{x}_{i}-
                    \boldsymbol{x}_{j}\right|}
        """
        self.betab = betab
        self.dim = dim
        self.fkern = fkern
        self.alphac = alphac
        self.gammam1 = gamma - 1.0
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs, d_au, d_av,
             d_aw, d_ae, XIJ, VIJ, DWI, DWJ, d_alpha, s_alpha, RIJ, d_h,
             d_dndh, d_n, s_h, s_dndh, s_n, d_e, s_e, d_dpsumdh, s_dpsumdh,
             RHOIJ1):

        dim = self.dim
        gammam1 = self.gammam1
        avi = declare("matrix(3)")

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        mj = s_m[s_idx]
        vijdotxij = (VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2])

        if RIJ < 1e-8:
            vs = 2 * cij
            muij = 0.0
            Fij = 0.0
        else:
            vs = 2 * cij - 3 * vijdotxij / RIJ
            muij = vijdotxij / RIJ
            Fij = 0.5 * (XIJ[0] * (DWI[0] + DWJ[0]) +
                         XIJ[1] * (DWI[1] + DWJ[1]) +
                         XIJ[2] * (DWI[2] + DWJ[2])) / RIJ

        # Artificial viscosity
        if vijdotxij <= 0.0:
            alphaij = 0.5 * (d_alpha[d_idx] + s_alpha[s_idx])
            oby2rhoij = RHOIJ1 / 2.0
            common = (alphaij * muij * (
                    cij - self.betab * muij) * mj * oby2rhoij)

            avi[0] = common * (DWI[0] + DWJ[0])
            avi[1] = common * (DWI[1] + DWJ[1])
            avi[2] = common * (DWI[2] + DWJ[2])

            # viscous contribution to velocity
            d_au[d_idx] += avi[0]
            d_av[d_idx] += avi[1]
            d_aw[d_idx] += avi[2]

            # viscous contribution to the thermal energy
            d_ae[d_idx] -= 0.5 * (VIJ[0] * avi[0] +
                                  VIJ[1] * avi[1] +
                                  VIJ[2] * avi[2])

            # artificial conductivity
            eij = d_e[d_idx] - s_e[s_idx]
            Lij = abs(d_p[d_idx] - s_p[s_idx]) / (d_p[d_idx] + s_p[s_idx])
            d_ae[d_idx] += (self.alphac * mj * alphaij * vs * eij * Lij * Fij *
                            oby2rhoij)

        # grad-h correction terms.
        hibynidim = d_h[d_idx] / (d_n[d_idx] * dim)
        inbrkti = 1 + d_dndh[d_idx] * hibynidim
        inprthsi = d_dpsumdh[d_idx] * hibynidim / (
                gammam1 * s_m[s_idx] * d_e[d_idx])
        fij = 1 - inprthsi / inbrkti

        hjbynjdim = s_h[s_idx] / (s_n[s_idx] * dim)
        inbrktj = 1 + s_dndh[s_idx] * hjbynjdim
        inprthsj = s_dpsumdh[s_idx] * hjbynjdim / (
                gammam1 * d_m[d_idx] * s_e[s_idx])
        fji = 1 - inprthsj / inbrktj

        # accelerations for velocity
        gammam1sq = gammam1 * gammam1
        comm = gammam1sq * mj * d_e[d_idx] * s_e[s_idx]
        commi = comm * fij / d_p[d_idx]
        commj = comm * fji / s_p[s_idx]

        d_au[d_idx] -= commi * DWI[0] + commj * DWJ[0]
        d_av[d_idx] -= commi * DWI[1] + commj * DWJ[1]
        d_aw[d_idx] -= commi * DWI[2] + commj * DWJ[2]

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]
        d_ae[d_idx] += commi * vijdotdwi


class WallBoundary(Equation):
    """
        :class:`WallBoundary
        <pysph.sph.gas_dynamics.boundary_equations.WallBoundary>` modified
        for PSPH
    """

    def initialize(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_h, d_htmp, d_h0,
                   d_u, d_v, d_w, d_wij, d_n, d_dndh, d_dpsumdh, d_m0):

        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_m0[d_idx] = d_m[d_idx]
        d_m[d_idx] = 0.0
        d_rho[d_idx] = 0.0
        d_e[d_idx] = 0.0
        d_cs[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_h[d_idx] = d_h0[d_idx]
        d_htmp[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0
        d_dpsumdh[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_e, d_m, d_cs, d_u, d_v, d_w,
             d_wij, d_htmp, s_p, s_rho, s_e, s_m, s_cs, s_h, s_u, s_v, s_w, WI,
             s_n, d_n, d_dndh, s_dndh, d_dpsumdh, s_dpsumdh):
        d_wij[d_idx] += WI
        d_p[d_idx] += s_p[s_idx] * WI
        d_u[d_idx] -= s_u[s_idx] * WI
        d_v[d_idx] -= s_v[s_idx] * WI
        d_w[d_idx] -= s_w[s_idx] * WI
        d_m[d_idx] += s_m[s_idx] * WI
        d_rho[d_idx] += s_rho[s_idx] * WI
        d_e[d_idx] += s_e[s_idx] * WI
        d_cs[d_idx] += s_cs[s_idx] * WI
        d_htmp[d_idx] += s_h[s_idx] * WI
        d_n[d_idx] += s_n[s_idx] * WI
        d_dndh[d_idx] += s_dndh[s_idx] * WI
        d_dpsumdh[d_idx] += s_dpsumdh[s_idx] * WI

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_h, d_u, d_v, d_w,
                  d_wij, d_htmp, d_dndh, d_dpsumdh, d_n, d_m0):
        if d_wij[d_idx] > 1e-30:
            d_p[d_idx] = d_p[d_idx] / d_wij[d_idx]
            d_u[d_idx] = d_u[d_idx] / d_wij[d_idx]
            d_v[d_idx] = d_v[d_idx] / d_wij[d_idx]
            d_w[d_idx] = d_w[d_idx] / d_wij[d_idx]
            d_m[d_idx] = d_m[d_idx] / d_wij[d_idx]
            d_rho[d_idx] = d_rho[d_idx] / d_wij[d_idx]
            d_e[d_idx] = d_e[d_idx] / d_wij[d_idx]
            d_cs[d_idx] = d_cs[d_idx] / d_wij[d_idx]
            d_h[d_idx] = d_htmp[d_idx] / d_wij[d_idx]
            d_dndh[d_idx] /= d_wij[d_idx]
            d_dpsumdh[d_idx] /= d_wij[d_idx]
            d_n[d_idx] /= d_wij[d_idx]

        # Secret Sauce
        if d_m[d_idx] < 1e-10:
            d_m[d_idx] = d_m0[d_idx]


class UpdateGhostProps(Equation):
    def __init__(self, dest, sources=None, dim=2):
        """
        :class:`MPMUpdateGhostProps
        <pysph.sph.gas_dynamics.basic.MPMUpdateGhostProps>` modified
        for PSPH
        """
        super().__init__(dest, sources)
        self.dim = dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_tag, d_h, d_rho, d_dndh,
                   d_psumdh, d_n):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_h[d_idx] = d_h[idx]
            d_rho[d_idx] = d_rho[idx]
            d_dndh[d_idx] = d_dndh[idx]
            d_psumdh[d_idx] = d_psumdh[idx]
            d_n[d_idx] = d_n[idx]


class PECStep(IntegratorStep):
    """Predictor Corrector integrator for Gas-dynamics modified for PSPH"""

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0,
                   d_v0, d_w0, d_u, d_v, d_w, d_e, d_e0, d_h0, d_converged,
                   d_rho, d_rho0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_e0[d_idx] = d_e[d_idx]

        d_h0[d_idx] = d_h[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]

        # set the converged attribute to 0 at the beginning of a Group
        d_converged[d_idx] = 0

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_e0, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_rho0,
               d_arho, d_h, d_h0, d_ah, dt):
        dtb2 = 0.5 * dt

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        # update thermal energy
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]

        # predict density and smoothing lengths for faster
        # convergence. NNPS need not be explicitly updated since it
        # will be called at the end of the predictor stage.
        d_h[d_idx] = d_h0[d_idx] + dtb2 * d_ah[d_idx]
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_e0, d_e, d_au, d_av, d_aw, d_ae, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_e[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]
