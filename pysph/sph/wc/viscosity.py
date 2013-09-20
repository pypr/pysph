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
                                       
        
