"""Basic WCSPH equations.
"""

from pysph.sph.equation import Equation


class ContinuityEquation(Equation):
    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ=[0.0, 0.0, 0.0],
             VIJ=[0.0, 0.0, 0.0]):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class SummationDensity(Equation):
    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ=0.0):
        d_rho[d_idx] += s_m[s_idx]*WIJ

class BodyForce(Equation):
    def __init__(self, dest, sources,
                 fx=0.0, fy=0.0, fz=0.0):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(BodyForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.fx
        d_av[d_idx] += self.fy
        d_aw[d_idx] += self.fz

class TaitEOS(Equation):
    def __init__(self, dest, sources=None,
                 rho0=1000.0, c0=1.0, gamma=7.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.B = rho0*c0*c0/gamma
        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.B * (tmp - 1.0)
        d_cs[d_idx] = self.c0 * pow( ratio, self.gamma1 )

class MomentumEquation(Equation):
    def __init__(self, dest, sources=None,
                 alpha=1.0, beta=1.0, eta=0.1, gx=0.0, gy=0.0, gz=0.0,
                 c0=1.0):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dt_fac = 0.0
        self.c0 = c0
        super(MomentumEquation, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ=[0.0, 0.0, 0.0],
             XIJ=[0.0, 0.0, 0.0], HIJ=1.0, R2IJ=1.0, RHOIJ1=1.0,
             DWIJ=[1.0, 1.0, 1.0], DT_ADAPT=[0.0, 0.0, 0.0]):
        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + self.eta*self.eta*HIJ*HIJ)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        # compute the CFL time step factor
        _dt_fac = 0.0
        if R2IJ > 1e-12:
            _dt_fac = abs( HIJ * vijdotxij/R2IJ ) + self.c0
            DT_ADAPT[0] = max(_dt_fac, DT_ADAPT[0])

        tmp = d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[2]


    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] +=  self.gx
        d_av[d_idx] +=  self.gy
        d_aw[d_idx] +=  self.gz


class XSPHCorrection(Equation):
    def __init__(self, dest, sources=None, eps=0.5):
        self.eps = eps
        super(XSPHCorrection, self).__init__(dest, sources)

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, d_az, WIJ=1.0, RHOIJ1=1.0,
             VIJ=[1.0,1,1]):
        tmp = -self.eps * s_m[s_idx]*WIJ*RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]
        d_az[d_idx] += tmp * VIJ[2]


    def post_loop(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]
        d_az[d_idx] += d_w[d_idx]
