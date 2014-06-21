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
    def initialize(self, d_idx, d_rho, d_div, d_grhox, d_grhoy, d_arho):
        d_rho[d_idx] = 0.0
        d_div[d_idx] = 0.0
        
        d_grhox[d_idx] = 0.0
        d_grhoy[d_idx] = 0.0
        d_arho[d_idx]  = 0.0
        
    def loop(self, d_idx, s_idx, d_rho, d_grhox, d_grhoy, d_arho, 
             s_m, VIJ, WI, DWI):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0]*DWI[0] + VIJ[1]*DWI[1] + VIJ[2]*DWI[2]
        
        # density
        d_rho[d_idx] += mj * WI
        
        # density accelerations
        d_arho[d_idx] += mj * vijdotdwij
        
        # gradient of density
        d_grhox[d_idx] += mj * DWI[0]
        d_grhoy[d_idx] += mj * DWI[1]

    def post_loop(self, d_idx, d_arho, d_rho, d_div):
        d_div[d_idx] = -d_arho[d_idx]/d_rho[d_idx]

class IdealGasEOS(Equation):
    def __init__(self, dest, sources=None, gamma=1.4):
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super(IdealGasEOS, self).__init__(dest, sources)
        
    def loop(self, d_idx, d_p, d_rho, d_e, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]
        if d_p[d_idx] < 0:
            print 'IDEALGASEOS', d_idx, self.gamma1, d_rho[d_idx], d_e[d_idx]

        d_cs[d_idx] = sqrt( self.gamma * d_p[d_idx]/d_rho[d_idx] )

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
