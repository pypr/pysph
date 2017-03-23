"""
Equations for the High Velocity Impact Problems
###############################################
"""

from math import sqrt
from pysph.sph.equation import Equation

class VonMisesPlasticity2D(Equation):
    def __init__(self, dest, sources, flow_stress):
        self.flow_stress2 = float(flow_stress*flow_stress)
        self.factor = sqrt( 2.0/3.0 )*flow_stress
        super(VonMisesPlasticity2D,self).__init__(dest, sources)

    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22):
        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        J = s00a* s00a + 2.0 * s01a*s10a + s11a*s11a
        scale = 1.0
        if (J > 2.0/3.0 * self.flow_stress2):
            scale = self.factor/sqrt(J)

        # store the stresses
        d_s00[d_idx] = scale * s00a
        d_s01[d_idx] = scale * s01a
        d_s02[d_idx] = scale * s02a

        d_s11[d_idx] = scale * s11a
        d_s12[d_idx] = scale * s12a

        d_s22[d_idx] = scale * s22a


class StiffenedGasEOS(Equation):

    """Stiffened-gas EOS from "A Free Lagrange Augmented Godunov Method for the
    Simulation of Elastic-Plastic Solids", B. P. Howell and G. J. Ball, JCP
    (2002).  http://dx.doi.org/10.1006/jcph.2001.6931
    """

    def __init__(self, dest, sources, gamma, r0, c0):
        self.gamma = float(gamma) # Gruneisen gamma
        self.c0    = float(c0)    # unshocked sound speed
        self.r0    = float(r0)    # reference density

        super(StiffenedGasEOS,self).__init__(dest, sources)

    def loop(self, d_idx, d_e, d_rho, d_p, d_cs):
        # Eq. (17)
        d_p[d_idx] = self.c0*self.c0 * (d_rho[d_idx] - self.r0) + \
                      (self.gamma - 1.0)*d_rho[d_idx]*d_e[d_idx]

        # Eq. (21)
        d_cs[d_idx] = sqrt(
            self.c0*self.c0 + (self.gamma - 1.0) *
            (d_e[d_idx] + d_p[d_idx]/d_rho[d_idx])
        )


class MieGruneisenEOS(Equation):
    def __init__(self, dest, sources, gamma,r0, c0, S):

        self.gamma = float(gamma)
        self.r0 = float(r0)
        self.c0 = float(c0)
        self.S = float(S)

        self.a0 = a0 = float(r0 * c0 * c0)
        self.b0 = a0 * ( 1 + 2.0*(S - 1.0) )
        self.c0 = a0 * ( 2*(S - 1.0) + 3*(S - 1.0)*(S - 1.0) )

        super(MieGruneisenEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho, d_e):
        rhoa = d_rho[d_idx]
        ea = d_e[d_idx]

        gamma = self.gamma
        ratio = rhoa/self.r0 - 1.0
        ratio2 = ratio * ratio

        PH = self.a0 * ratio
        if ratio > 0:
            PH = PH + ratio2 * (self.b0 + self.c0*ratio)

        d_p[d_idx] = (1. - 0.5*gamma*ratio) * PH + rhoa * ea * gamma
