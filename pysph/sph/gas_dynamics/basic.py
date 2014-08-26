"""Basic equations for Gas-dynamics"""

from pysph.sph.equation import Equation
from math import sqrt

class ScaleSmoothingLength(Equation):
    def __init__(self, dest, sources=None, factor=2.0):
        super(ScaleSmoothingLength, self).__init__(dest, sources)
        self.factor = factor
        
    def loop(self, d_idx, d_h):
        d_h[d_idx] = d_h[d_idx] * self.factor

class UpdateSmoothingLengthFromVolume(Equation):
    def __init__(self, dest, sources=None, k=1.2, dim=1.0):
        super(UpdateSmoothingLengthFromVolume, self).__init__(dest, sources)
        self.k = k
        self.dim1 = 1./dim
        
    def loop(self, d_idx, d_m, d_rho, d_h):
        d_h[d_idx] = self.k * pow( d_m[d_idx]/d_rho[d_idx], self.dim1)

class SummationDensity(Equation):
    def __init__(
        self, dest, sources=None, dim=2, density_iterations=False, iterate_only_once=False,
        k=1.2, htol=1e-6):

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
        
    def initialize(
        self, d_idx, d_rho, d_div, d_grhox, d_grhoy, d_arho, d_omega):
        d_rho[d_idx] = 0.0
        d_div[d_idx] = 0.0
        
        d_grhox[d_idx] = 0.0
        d_grhoy[d_idx] = 0.0
        d_arho[d_idx]  = 0.0

        # Set the default omega to 1.0.
        d_omega[d_idx] = 1.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_grhox, d_grhoy, d_arho, 
             d_dwdh, s_m, VIJ, WI, DWI, GHI):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]
        
        # density
        d_rho[d_idx] += mj * WI
        
        # density accelerations
        d_arho[d_idx] += mj * vijdotdwij
        
        # gradient of density
        d_grhox[d_idx] += mj * DWI[0]
        d_grhoy[d_idx] += mj * DWI[1]

        # gradient of kernel w.r.t h
        d_dwdh[d_idx] += mj * GHI

    def post_loop(self, d_idx, d_arho, d_rho, d_div, d_omega, d_dwdh,
                  d_h0, d_h, d_m, d_ah, d_converged):

        # iteratively find smoothing length consistent with the
        if self.density_iterations:
            if not ( d_converged[d_idx] == 1 ):
                # current mass and smoothing length. The initial
                # smoothing length h0 for this particle must be set
                # outside the Group (that is, in the integrator)
                mi = d_m[d_idx]; hi = d_h[d_idx]; hi0 = d_h0[d_idx]
                
                # density from the mass, smoothing length and kernel
                # scale factor
                rhoi = mi/(hi/self.k)**self.dim

                dhdrhoi = -hi/( self.dim*d_rho[d_idx] )
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
                if ( hnew > 1.2 * hi ):
                    hnew = 1.2 * hi
                elif ( hnew < 0.8 * hi ):
                    hnew = 0.8 * hi
                
                # overwrite if gone awry
                if ( (hnew <= 1e-6) or (gradhi < 1e-6) ):
                    hnew = self.k * (mi/d_rho[d_idx])**(1./self.dim)

                # check for convergence
                diff = abs( hnew-hi )/hi0
                
                if not ( (diff < self.htol) and (omegai > 0) or self.iterate_only_once):
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
    def __init__(self, dest, sources=None, gamma=1.4):
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super(IdealGasEOS, self).__init__(dest, sources)
        
    def loop(self, d_idx, d_p, d_rho, d_e, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]
        d_cs[d_idx] = sqrt( self.gamma * d_p[d_idx]/d_rho[d_idx] )

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

class MPMAccelerations(Equation):
    def __init__(self, dest, sources, alpha1=1.0, alpha2=0.1, beta=2.0):
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.beta = beta
        super(MPMAccelerations, self).__init__(dest, sources)
        
    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_am):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs,
             d_e, s_e, d_rho, s_rho, d_au, d_av, d_aw, d_ae,
             XIJ, VIJ, DWI, DWJ, DWIJ, HIJ, EPS, RIJ, R2IJ, RHOIJ,
             DT_ADAPT):

        # particle pressure
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        # pi/rhoi**2
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        pibrhoi2 = pi/rhoi2
        
        # pj/rhoj**2
        rhoj2 = s_rho[s_idx]*s_rho[s_idx]
        pjbrhoj2 = pj/rhoj2
        
        # averaged mass
        mi = d_m[d_idx]
        mj = s_m[s_idx]
        mij = 0.5 * (mi + mj)

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

        # v_{ij} \cdot r_{ij}
        dot = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        
        # scalar part of the kernel gradient DWIJ
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # signal velocities
        pdiff = abs(pi-pj)
        vsig1 = 0.5 * max(ci + cj - self.beta*dot, 0.0)
        vsig2 = sqrt( pdiff/RHOIJ )

        # Artificial viscosity
        if dot <= 0.0:

            # viscosity
            tmpv = mj/RHOIJ * self.alpha1 * vsig1 * dot
            d_au[d_idx] += tmpv * DWIJ[0]
            d_av[d_idx] += tmpv * DWIJ[1]
            d_aw[d_idx] += tmpv * DWIJ[2]

            # viscous contribution to the thermal energy
            d_ae[d_idx] += -0.5*mj/RHOIJ*self.alpha1*vsig1*dot*dot*Fij

        # gradient terms
        d_au[d_idx] += -mj*(pibrhoi2*DWI[0] + pjbrhoj2*DWJ[0])
        d_av[d_idx] += -mj*(pibrhoi2*DWI[1] + pjbrhoj2*DWJ[1])
        d_aw[d_idx] += -mj*(pibrhoi2*DWI[2] + pjbrhoj2*DWJ[2])

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * vijdotdwi

        # thermal conduction
        eij = d_e[d_idx] - s_e[s_idx]
        d_ae[d_idx] += mj/RHOIJ * self.alpha2 * vsig2 * eij * Fij
