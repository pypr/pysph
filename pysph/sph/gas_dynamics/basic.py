"""Basic equations for Gas-dynamics"""

from compyle.api import declare
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from pysph.sph.equation import Equation
from math import sqrt, exp, log
from pysph.base.particle_array import get_ghost_tag
import numpy

GHOST_TAG = get_ghost_tag()


class ScaleSmoothingLength(Equation):
    def __init__(self, dest, sources, factor=2.0):
        super(ScaleSmoothingLength, self).__init__(dest, sources)
        self.factor = factor

    def loop(self, d_idx, d_h):
        d_h[d_idx] = d_h[d_idx] * self.factor


class UpdateSmoothingLengthFromVolume(Equation):
    def __init__(self, dest, sources, dim, k=1.2):
        super(UpdateSmoothingLengthFromVolume, self).__init__(dest, sources)
        self.k = k
        self.dim1 = 1./dim

    def loop(self, d_idx, d_m, d_rho, d_h):
        d_h[d_idx] = self.k * pow(d_m[d_idx]/d_rho[d_idx], self.dim1)


class SummationDensityADKE(Equation):
    """
    References
    ----------
    ..  A comparison of SPH schemes for the compressible Euler equations,
        2014, Journal of Computational Physics, 256, pp 308 -- 333
            (http://dx.doi.org/10.1016/j.jcp.2013.08.060)
    """
    def __init__(self, dest, sources, k=1.0,  eps=0.0):
        self.k = k
        self.eps = eps
        super(SummationDensityADKE, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho, d_rho, d_h, d_h0):
        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0
        d_h[d_idx] = d_h0[d_idx]

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_m,  VIJ, DWI, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ
        mj = s_m[s_idx]
        vijdotdwij = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]

        # density accelerations
        d_arho[d_idx] += mj * vijdotdwij

    def post_loop(self, d_idx, d_rho, d_arho, d_div, d_logrho):
        d_div[d_idx] = -d_arho[d_idx]/d_rho[d_idx]
        d_arho[d_idx] = 0
        d_logrho[d_idx] = log(d_rho[d_idx])

    def reduce(self, dst, t, dt):
        n = len(dst.x)
        tmp_sum_logrho = serial_reduce_array(dst.logrho, 'sum')
        sum_logrho = parallel_reduce_array(tmp_sum_logrho, 'sum')
        g = exp(sum_logrho/n)

        lamda = declare('object')
        lamda = self.k*numpy.power(g/dst.rho, self.eps)
        dst.h[:] = lamda*dst.h0


class SummationDensity(Equation):
    def __init__(self, dest, sources, dim, density_iterations=False,
                 iterate_only_once=False, k=1.2, htol=1e-6):
        r"""Summation density with iterative solution of the smoothing lengths.

        Parameters:

        density_iterations : bint
            Flag to indicate density iterations are required.

        iterate_only_once : bint
            Flag to indicate if only one iteration is required

        k : double
            Kernel scaling factor

        htol : double
            Iteration tolerance

        """
        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.k = k
        self.htol = htol

        # by default, we set the equation_has_converged attribute to True. If
        # density_iterations is set to True, we will have at least one
        # iteration to determine the new smoothing lengths since the
        # 'converged' property of the particles is intialized to False
        self.equation_has_converged = 1

        super(SummationDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_div, d_grhox, d_grhoy, d_grhoz,
                   d_arho, d_dwdh):

        d_rho[d_idx] = 0.0
        d_div[d_idx] = 0.0

        d_grhox[d_idx] = 0.0
        d_grhoy[d_idx] = 0.0
        d_grhoz[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_dwdh[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_grhox, d_grhoy, d_grhoz, d_arho,
             d_dwdh, s_m, d_converged, VIJ, WI, DWI, GHI):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]

        # density
        d_rho[d_idx] += mj * WI

        # density accelerations
        d_arho[d_idx] += mj * vijdotdwij

        # gradient of density
        d_grhox[d_idx] += mj * DWI[0]
        d_grhoy[d_idx] += mj * DWI[1]
        d_grhoz[d_idx] += mj * DWI[2]

        # gradient of kernel w.r.t h
        d_dwdh[d_idx] += mj * GHI

    def post_loop(self, d_idx, d_arho, d_rho, d_div, d_omega, d_dwdh,
                  d_h0, d_h, d_m, d_ah, d_converged):

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
                rhoi = mi/(hi/self.k)**self.dim

                dhdrhoi = -hi/(self.dim*d_rho[d_idx])
                dwdhi = d_dwdh[d_idx]
                omegai = 1.0 - dhdrhoi*dwdhi

                # correct omegai
                if omegai < 0:
                    omegai = 1.0

                # kernel multiplier. These are the multiplicative
                # pre-factors, or the "grah-h" terms in the
                # equations. Remember that the equations use 1/omega
                gradhi = 1.0/omegai
                d_omega[d_idx] = gradhi

                # the non-linear function and it's derivative
                func = rhoi - d_rho[d_idx]
                dfdh = omegai/dhdrhoi

                # Newton Raphson estimate for the new h
                hnew = hi - func/dfdh

                # Nanny control for h
                if (hnew > 1.2 * hi):
                    hnew = 1.2 * hi
                elif (hnew < 0.8 * hi):
                    hnew = 0.8 * hi

                # overwrite if gone awry
                if ((hnew <= 1e-6) or (gradhi < 1e-6)):
                    hnew = self.k * (mi/d_rho[d_idx])**(1./self.dim)

                # check for convergence
                diff = abs(hnew - hi)/hi0

                if not ((diff < self.htol) and (omegai > 0) or
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
                    d_arho[d_idx] *= d_omega[d_idx]
                    d_ah[d_idx] = d_arho[d_idx] * dhdrhoi
                    d_converged[d_idx] = 1

        # comptue the divergence of velocity
        d_div[d_idx] = -d_arho[d_idx]/d_rho[d_idx]

    def converged(self):
        return self.equation_has_converged


class IdealGasEOS(Equation):
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super(IdealGasEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho, d_e, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]
        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx]/d_rho[d_idx])


class Monaghan92Accelerations(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

        super(Monaghan92Accelerations, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_p, s_p, d_cs, s_cs,
             d_au, d_av, d_aw, d_ae, s_m,
             VIJ, DWIJ, XIJ, EPS, HIJ, R2IJ, RHOIJ1):

        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        tmpi = d_p[d_idx]/rhoi2
        tmpj = s_p[s_idx]/rhoj2

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        piij = 0.0
        if vijdotxij < 0:
            muij = HIJ*vijdotxij/(R2IJ + EPS)
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij *= RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * (tmpi + tmpj + piij) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmpi + tmpj + piij) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmpi + tmpj + piij) * DWIJ[2]

        vijdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]

        d_ae[d_idx] += 0.5 * s_m[s_idx] * (tmpi + tmpj + piij) * vijdotdwij


class ADKEAccelerations(Equation):
    """
    Reference
    ---------
    ..  A comparison of SPH schemes for the compressible Euler equations,
        2014, Journal of Computational Physics, 256, pp 308 -- 333
            (http://dx.doi.org/10.1016/j.jcp.2013.08.060)
    """
    def __init__(self, dest, sources, alpha, beta, g1, g2, k, eps):
        self.alpha = alpha
        self.g1 = g1
        self.g2 = g1
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.eps = eps
        super(ADKEAccelerations, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):

        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, d_ae, d_p, s_p, d_rho,
             s_rho, d_m, s_m, d_cs, s_cs, s_e, d_e, s_h, d_h, s_div, d_div,
             DWIJ, HIJ, XIJ, VIJ, R2IJ, EPS, RHOIJ, RHOIJ1):

        # particle pressure
        p_i = d_p[d_idx]
        pj = s_p[s_idx]
        # p_i/rhoi**2
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        pibrhoi2 = p_i/rhoi2

        # pj/rhoj**2
        rhoj2 = s_rho[s_idx]*s_rho[s_idx]
        pjbrhoj2 = pj/rhoj2

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        # averaged mass
        mj = s_m[s_idx]

        # averaged sound speed
        ci = d_cs[d_idx]
        cj = s_cs[s_idx]
        cij = 0.5 * (ci + cj)

        hi = d_h[d_idx]
        hj = s_h[s_idx]

        divi = d_div[d_idx]
        divj = s_div[s_idx]

        ei = d_e[d_idx]
        ej = s_e[s_idx]

        # Themal Conduction
        Hi = self.g1 * hi * ci + self.g2 * hi * hi*(abs(divi)-divi)
        Hj = self.g1 * hj * cj + self.g2 * hj * hj*(abs(divj)-divj)
        Hij = (Hi+Hj)*(ei-ej)/(RHOIJ*(R2IJ+EPS))

        xijdotvij = XIJ[0]*VIJ[0] + XIJ[1]*VIJ[1] + XIJ[2]*VIJ[2]
        piij = 0.0
        if xijdotvij < 0:
            muij = HIJ*xijdotvij/(R2IJ+EPS)
            piij = muij * (self.beta*muij - self.alpha*cij)*RHOIJ1
        tmpv = pibrhoi2 + pjbrhoj2 + piij
        d_au[d_idx] += -mj*tmpv * DWIJ[0]
        d_av[d_idx] += -mj*tmpv * DWIJ[1]
        d_aw[d_idx] += -mj*tmpv * DWIJ[2]
        vijdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        d_ae[d_idx] += 0.5*mj*(tmpv*vijdotdwij + 2*xijdotdwij*Hij)


class MPMAccelerations(Equation):
    def __init__(self, dest, sources, beta=2.0, update_alpha1=False,
                 update_alpha2=False, alpha1_min=0.1, alpha2_min=0.1,
                 sigma=0.1):
        self.beta = beta
        self.sigma = sigma

        self.update_alpha1 = update_alpha1
        self.update_alpha2 = update_alpha2

        self.alpha1_min = alpha1_min
        self.alpha2_min = alpha2_min

        super(MPMAccelerations, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_am,
                   d_aalpha1, d_aalpha2, d_del2e, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

        d_aalpha1[d_idx] = 0.0
        d_aalpha2[d_idx] = 0.0

        d_del2e[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs,
             d_e, s_e, d_rho, s_rho, d_au, d_av, d_aw, d_ae,
             d_omega, s_omega, XIJ, VIJ, DWI, DWJ, DWIJ, HIJ,
             d_del2e, d_alpha1, s_alpha1, d_alpha2, s_alpha2,
             EPS, RIJ, R2IJ, RHOIJ, d_dt_cfl):

        # particle pressure
        p_i = d_p[d_idx]
        pj = s_p[s_idx]

        # p_i/rhoi**2
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        pibrhoi2 = p_i/rhoi2

        # pj/rhoj**2
        rhoj2 = s_rho[s_idx]*s_rho[s_idx]
        pjbrhoj2 = pj/rhoj2

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        mj = s_m[s_idx]

        # averaged sound speed
        ci = d_cs[d_idx]
        cj = s_cs[s_idx]
        cij = 0.5 * (ci + cj)

        # normalized interaction vector
        if RIJ < 1e-8:
            XIJ[0] = 0.0
            XIJ[1] = 0.0
            XIJ[2] = 0.0
        else:
            XIJ[0] /= RIJ
            XIJ[1] /= RIJ
            XIJ[2] /= RIJ

        # v_{ij} \cdot r_{ij} or vijdotxij
        dot = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        # scalar part of the kernel gradient DWIJ
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # signal velocities
        pdiff = abs(p_i - pj)
        vsig1 = 0.5 * max(2*cij - self.beta*dot, 0.0)
        vsig2 = sqrt(pdiff/RHOIJ)

        # compute the Courant-limited time step factor.
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], cij + self.beta * dot)

        # Artificial viscosity
        if dot <= 0.0:

            # viscosity
            alpha1 = 0.5 * (d_alpha1[d_idx] + s_alpha1[s_idx])
            tmpv = mj/RHOIJ * alpha1 * vsig1 * dot
            d_au[d_idx] += tmpv * DWIJ[0]
            d_av[d_idx] += tmpv * DWIJ[1]
            d_aw[d_idx] += tmpv * DWIJ[2]

            # viscous contribution to the thermal energy
            d_ae[d_idx] += -0.5*mj/RHOIJ*alpha1*vsig1*dot*dot*Fij

        # grad-h correction terms. These will be set to 1.0 by the
        # integrator and thus can be used safely.
        omegai = d_omega[d_idx]
        omegaj = s_omega[s_idx]

        # gradient terms
        d_au[d_idx] += -mj*(pibrhoi2*omegai*DWI[0] + pjbrhoj2*omegaj*DWJ[0])
        d_av[d_idx] += -mj*(pibrhoi2*omegai*DWI[1] + pjbrhoj2*omegaj*DWJ[1])
        d_aw[d_idx] += -mj*(pibrhoi2*omegai*DWI[2] + pjbrhoj2*omegaj*DWJ[2])

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * omegai * vijdotdwi

        # thermal conduction
        alpha2 = 0.5 * (d_alpha2[d_idx] + s_alpha2[s_idx])
        eij = d_e[d_idx] - s_e[s_idx]
        d_ae[d_idx] += mj/RHOIJ * alpha2 * vsig2 * eij * Fij

        # Laplacian of thermal energy
        d_del2e[d_idx] += mj/s_rho[s_idx] * eij/(RIJ + EPS) * Fij

    def post_loop(self, d_idx, d_h, d_cs, d_alpha1, d_aalpha1, d_div,
                  d_del2e, d_e, d_alpha2, d_aalpha2):

        hi = d_h[d_idx]
        tau = hi/(self.sigma*d_cs[d_idx])

        if self.update_alpha1:
            S1 = max(-d_div[d_idx], 0.0)
            d_aalpha1[d_idx] = (self.alpha1_min - d_alpha1[d_idx])/tau + S1

        if self.update_alpha2:
            S2 = 0.01 * d_h[d_idx] * abs(d_del2e[d_idx])/sqrt(d_e[d_idx])
            d_aalpha2[d_idx] = (self.alpha2_min - d_alpha2[d_idx])/tau + S2


class MPMUpdateGhostProps(Equation):
    def __init__(self, dest, sources=None, dim=2):
        super(MPMUpdateGhostProps, self).__init__(dest, sources)
        self.dim = dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_cs, d_tag):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_cs[d_idx] = d_cs[idx]


class ADKEUpdateGhostProps(Equation):
    def __init__(self, dest, sources=None, dim=2):
        super(ADKEUpdateGhostProps, self).__init__(dest, sources)
        self.dim = dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_cs, d_tag, d_rho):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_cs[d_idx] = d_cs[idx]
            d_rho[d_idx] = d_rho[idx]
