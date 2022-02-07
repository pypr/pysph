from compyle.api import declare
from pysph.base.particle_array import get_ghost_tag

from pysph.sph.equation import Equation
from pysph.sph.wc.linalg import identity, gj_solve, augmented_matrix, mat_mult

GHOST_TAG = get_ghost_tag()


class SummationDensity(Equation):
    def __init__(self, dest, sources, dim, density_iterations=False,
                 iterate_only_once=False, k=1.2, htol=1e-6):

        r"""Summation density with iterative solution of the smoothing lengths
        from pysph.sph.gas_dynamics.basic modified to use number density
        for calculation of grad-h terms.
        """

        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.k = k
        self.htol = htol
        self.equation_has_converged = 1

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_drhosumdh, d_n, d_dndh,
                   d_prevn, d_prevdndh, d_prevdrhosumdh):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdrhosumdh[d_idx] = d_drhosumdh[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]

        d_drhosumdh[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, d_drhosumdh, s_m, VIJ, WI, DWI,
             GHI, d_n, d_dndh, d_h, d_prevn, d_prevdndh, d_prevdrhosumdh):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        d_rho[d_idx] += mj * WI

        # density accelerations
        hibynidim = d_h[d_idx] / (d_prevn[d_idx] * self.dim)
        inbrkti = 1 + d_prevdndh[d_idx] * d_h[d_idx] * hibynidim
        inprthsi = d_prevdrhosumdh[d_idx] * hibynidim
        fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)
        d_arho[d_idx] += mj * vijdotdwij * fij

        # gradient of kernel w.r.t h
        d_drhosumdh[d_idx] += mj * GHI
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_arho, d_rho, d_drhosumdh, d_h0, d_h, d_m,
                  d_ah, d_converged):

        # iteratively find smoothing length consistent with the
        if self.density_iterations:
            if not (d_converged[d_idx] == 1):
                # current mass and smoothing length. The initial
                # smoothing length h0 for this particle must be set
                # outside the Group (that is, in the integrator)
                mi = d_m[d_idx]
                hi = d_h[d_idx]
                hi0 = d_h0[d_idx]

                # density from the mass, smoothing length and kernel
                # scale factor
                rhoi = mi / (hi / self.k) ** self.dim

                # using fi from density entropy formulation for convergence
                # related checks.
                dhdrhoi = -hi / (self.dim * d_rho[d_idx])
                obyfi = 1.0 - dhdrhoi * d_drhosumdh[d_idx]

                # correct fi TODO: Remove if not required
                if obyfi < 0:
                    obyfi = 1.0

                # kernel multiplier. These are the multiplicative
                # pre-factors, or the "grah-h" terms in the
                # equations. Remember that the equations use 1/omega
                fi = 1.0 / obyfi

                # the non-linear function and it's derivative
                func = d_rho[d_idx] - rhoi
                dfdh = d_drhosumdh[d_idx] - 1 / dhdrhoi

                # Newton Raphson estimate for the new h
                hnew = hi - func / dfdh

                # Nanny control for h TODO: Remove if not required
                if (hnew > 1.2 * hi):
                    hnew = 1.2 * hi
                elif (hnew < 0.8 * hi):
                    hnew = 0.8 * hi

                # overwrite if gone awry TODO: Remove if not required
                if ((hnew <= 1e-6) or (fi < 1e-6)):
                    hnew = self.k * (mi / d_rho[d_idx]) ** (1. / self.dim)

                # check for convergence
                diff = abs(hnew - hi) / hi0

                if not ((diff < self.htol) and (obyfi > 0) or
                        self.iterate_only_once):
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
                    d_ah[d_idx] = d_arho[d_idx] * dhdrhoi
                    d_converged[d_idx] = 1

    def converged(self):
        return self.equation_has_converged

class TSPHMomentumAndEnergy(Equation):
    def __init__(self, dest, sources, dim, fkern, beta=2.0):
        self.beta = beta
        self.dim = dim
        self.fkern = fkern
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho,
             d_au, d_av, d_aw, d_ae, XIJ, VIJ, DWI, DWJ, HIJ, d_alpha1,
             s_alpha1, RIJ, R2IJ, RHOIJ, d_dt_cfl, d_h, d_dndh, d_n,
             d_drhosumdh, s_h, s_dndh, s_n, s_drhosumdh):

        dim = self.dim

        # particle pressure
        p_i = d_p[d_idx]
        pj = s_p[s_idx]

        # p_i/rhoi**2
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        pibrhoi2 = p_i / rhoi2

        # pj/rhoj**2
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]
        pjbrhoj2 = pj / rhoj2

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        mj = s_m[s_idx]
        hij = self.fkern * HIJ
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        # normalized interaction vector
        if RIJ < 1e-8:
            XIJ[0] = 0.0
            XIJ[1] = 0.0
            XIJ[2] = 0.0
        else:
            XIJ[0] /= RIJ
            XIJ[1] /= RIJ
            XIJ[2] /= RIJ

        # Is this really reqd?
        # # v_{ij} \cdot r_{ij} or vijdotxij
        # dot = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]
        #
        # # compute the Courant-limited time step factor.
        # d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], cij + self.beta * dot)

        # Artificial viscosity
        if vijdotxij <= 0.0:
            # viscosity
            alpha1 = 0.5 * (d_alpha1[d_idx] + s_alpha1[s_idx])
            muij = hij * vijdotxij / (R2IJ + 0.0001 * hij ** 2)
            common = alpha1 * muij * (cij - 2 * muij) * mj / (2 * RHOIJ)

            aaui = common * (DWI[0] + DWJ[0])
            aavi = common * (DWI[1] + DWJ[1])
            aawi = common * (DWI[2] + DWJ[2])
            d_au[d_idx] += aaui
            d_av[d_idx] += aavi
            d_aw[d_idx] += aawi

            # viscous contribution to the thermal energy
            d_ae[d_idx] -= 0.5 * (
                    VIJ[0] * aaui + VIJ[1] * aavi + VIJ[2] * aawi)

        # grad-h correction terms.
        hibynidim = d_h[d_idx] / (d_n[d_idx] * dim)
        inbrkti = 1 + d_dndh[d_idx] * d_h[d_idx] * hibynidim
        inprthsi = d_drhosumdh[d_idx] * hibynidim
        fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)

        hjbynjdim = s_h[s_idx] / (s_n[s_idx] * dim)
        inbrktj = 1 + s_dndh[s_idx] * s_h[s_idx] * hjbynjdim
        inprthsj = s_drhosumdh[s_idx] * hibynidim
        fji = 1 - inprthsj / (d_m[d_idx] * inbrktj)

        # accelerations for velocity
        mj_pibrhoi_fij = mj * pibrhoi2 * fij
        mj_pjbrhoj_fji = mj * pjbrhoj2 * fji

        d_au[d_idx] -= mj_pibrhoi_fij * DWI[0] + mj_pjbrhoj_fji * DWJ[0]
        d_av[d_idx] -= mj_pibrhoi_fij * DWI[1] + mj_pjbrhoj_fji * DWJ[1]
        d_aw[d_idx] -= mj_pibrhoi_fij * DWI[2] + mj_pjbrhoj_fji * DWJ[2]

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * fij * vijdotdwi


class MorrisMonaghanSwitch(Equation):
    def __init__(self, dest, sources, alpha1_min=0.1, sigma=0.1):
        self.sigma = sigma
        self.alpha1_min = alpha1_min
        super().__init__(dest, sources)

    def post_loop(self, d_h, d_idx, d_cs, d_divv, d_aalpha1, d_alpha1):
        hi = d_h[d_idx]
        tau = hi / (self.sigma * d_cs[d_idx])

        S1 = max(-d_divv[d_idx], 0.0)
        d_aalpha1[d_idx] = (self.alpha1_min - d_alpha1[d_idx]) / tau + S1


class BalsaraSwitch(Equation):
    def __init__(self, dest, sources, alphaav, fkern):
        self.alphaav = alphaav
        self.fkern = fkern
        super().__init__(dest, sources)

    def post_loop(self, d_h, d_idx, d_cs, d_divv, d_gradv, d_alpha1, dt,
                  d_aalpha1, d_alpha10, d_arho, d_rho):
        curlv = declare('matrix(3)')

        curlv[0] = (d_gradv[9 * d_idx + 3 * 2 + 1] -
                    d_gradv[9 * d_idx + 3 * 1 + 2])
        curlv[1] = (d_gradv[9 * d_idx + 3 * 0 + 2] -
                    d_gradv[9 * d_idx + 3 * 2 + 0])
        curlv[2] = (d_gradv[9 * d_idx + 3 * 1 + 0] -
                    d_gradv[9 * d_idx + 3 * 0 + 1])

        abscurlv = sqrt(curlv[0] * curlv[0] +
                        curlv[1] * curlv[1] +
                        curlv[2] * curlv[2])

        absdivv = abs(d_divv[d_idx])

        fhi = d_h[d_idx] * self.fkern

        d_alpha1[d_idx] = self.alphaav * absdivv / (
                absdivv + abscurlv + 0.0001 * d_cs[d_idx] / 1e-3)

        # Just to keep stage 2 of PEC happy
        # ---------------------------------
        # stage 2 of GasDFluidStep does,
        # d_alpha1[d_idx] = d_alpha10[d_idx] + dt*d_aalpha1[d_idx]
        # d_aalpha1 is already 0. d_alpha10[d_idx] is also 0 as the
        # integrator's initialise is before compute_accelerations.
        # So, d_alpha1[d_idx] is always rewritten to zero. To avoid this,
        d_alpha10[d_idx] = d_alpha1[d_idx]


class VelocityGradDivC1(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, mat_mult]

    def initialize(self, d_gradv, d_idx, d_invtt, d_divv):
        start_indx, i, dim = declare('int', 3)
        start_indx = 9 * d_idx
        for i in range(9):
            d_gradv[start_indx + i] = 0.0
            d_invtt[start_indx + i] = 0.0
        d_divv[d_idx] = 0.0

    def loop(self, d_idx, d_invtt, s_m, s_idx, VIJ, DWI, XIJ, d_gradv):
        start_indx, row, col, drowcol, dim = declare('int', 5)
        dim = self.dim
        start_indx = d_idx * 9
        for row in range(dim):
            for col in range(dim):
                drowcol = start_indx + row * 3 + col
                d_invtt[drowcol] -= s_m[s_idx] * XIJ[row] * DWI[col]
                d_gradv[drowcol] -= s_m[s_idx] * VIJ[row] * DWI[col]

    def post_loop(self, d_idx, d_gradv, d_invtt, d_divv):
        tt = declare('matrix(9)')
        invtt = declare('matrix(9)')
        augtt = declare('matrix(18)')
        idmat = declare('matrix(9)')
        gradv = declare('matrix(9)')
        start_indx, row, col, rowcol, drowcol, dim = declare('int', 6)
        dim = self.dim
        start_indx = 9 * d_idx
        identity(idmat, 3)
        identity(tt, 3)

        for row in range(3):
            for col in range(3):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                gradv[rowcol] = d_gradv[drowcol]

        for row in range(dim):
            for col in range(dim):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                tt[rowcol] = d_invtt[drowcol]

        augmented_matrix(tt, idmat, 3, 3, 3, augtt)
        gj_solve(augtt, 3, 3, invtt)
        gradvls = declare('matrix(9)')
        mat_mult(gradv, invtt, 3, gradvls)

        for row in range(dim):
            d_divv[d_idx] += gradvls[row * 3 + row]
            for col in range(dim):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                d_gradv[drowcol] = gradvls[rowcol]