"""Viscosity functions"""

from pysph.sph.equation import Equation

class LaminarViscosity(Equation):
    def __init__(self, dest, sources=None, nu=1e-6, eta=0.01):
        self.nu = nu
        self.eta = eta
        super(LaminarViscosity,self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho,
             d_au, d_av, d_aw,
             DWIJ=[0.0, 0.0, 0.0], XIJ=[0.0, 0.0, 0.0], 
             VIJ=[0.0, 0.0, 0.0], R2IJ=1.0, HIJ=1.0):
        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        
        mb = s_m[s_idx]
        
        tmp = mb * 4 * self.nu * Fij/( (rhoa + rhob)*(R2IJ + self.eta*HIJ*HIJ) )

        # accelerations
        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]

class MonaghanSignalViscosityFluids(Equation):
    def __init__(self, dest, sources=None, alpha=0.5, h=None):
        self.alpha=0.125 * alpha * h
        if h is None:
            raise ValueError("Invalid value for parameter h : %s"%h)
        super(MonaghanSignalViscosityFluids,self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             d_au, d_av, d_aw, d_cs, s_cs,
             RIJ=0.0, HIJ=0.0, VIJ=[0.0, 0.0, 0.0],
             XIJ=[0.0, 0.0, 0.0],
             DWIJ=[0.0, 0.0, 0.0]):

        nua = self.alpha * d_cs[d_idx]
        nub = self.alpha * s_cs[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        mb = s_m[s_idx]
        
        vabdotrab = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        force = -16 * nua * nub/(nua*rhoa + nub*rhob) * vabdotrab/(HIJ * (RIJ + 0.01*HIJ*HIJ))
        
        d_au[d_idx] += -mb * force * DWIJ[0]
        d_av[d_idx] += -mb * force * DWIJ[1]
        d_aw[d_idx] += -mb * force * DWIJ[2]
