"""Functions for the transport velocity formulation of Adami et. al.
"""

from pysph.sph.equation import Equation

class DensitySummation(Equation):
    def initialize(self, d_idx, d_V, d_rho):
        d_V[d_idx] = 0.0
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += d_m[d_idx]*WIJ

class VolumeSummation(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

class ContinuityEquation(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, d_m, s_V, VIJ, DWIJ):
        Vj = 1./s_V[s_idx]
        vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]
        d_arho[d_idx] += d_m[d_idx] * Vj * vijdotdwij

class StateEquation(Equation):
    def __init__(self, dest, sources=None, p0=1.0, rho0=1.0, b=1.0):
        self.b=b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * ( d_rho[d_idx]/self.rho0 - self.b )

class SolidWallBC(Equation):
    def __init__(self, dest, sources=None, rho0=1.0, p0=100.0,
                 gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=0.0,
                 b=1.0):
        self.rho0 = rho0
        self.p0 = p0
        self.b=b
        self.gx = gx; self.ax = ax
        self.gy = gy; self.ay = ay
        self.gz = gz; self.az = az
        super(SolidWallBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_wij):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_p, d_wij, s_u, s_v, s_p, s_rho,
             WIJ, XIJ):
        # smooth velocities at the ghost points
        d_u[d_idx] += s_u[s_idx]*WIJ
        d_v[d_idx] += s_v[s_idx]*WIJ

        # smooth pressure
        gdotxij = (self.gx-self.ax)*XIJ[0] + \
            (self.gy-self.ay)*XIJ[1] + \
            (self.gz-self.az)*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx] * gdotxij * WIJ

        # denominator
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_u, d_v, d_u0, d_v0, d_p, d_rho):
        # smooth velocity at the wall particle
        if d_wij[d_idx] > 1e-14:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]

            # solid wall condition
            d_u[d_idx] = 2*d_u0[d_idx] - d_u[d_idx]
            d_v[d_idx] = 2*d_v0[d_idx] - d_v[d_idx]

            # smooth interpolated pressure at the wall
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure
        d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + self.b)

class MomentumEquation(Equation):
    def __init__(self, dest, sources=None, pb=0.0, nu=0.01, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.nu = nu
        self.pb = pb
        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_auhat, d_avhat):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, d_u, d_v, d_V, d_au, d_av,
             d_auhat, d_avhat, d_m, d_uhat, d_vhat,
             s_rho, s_p, s_u, s_v, s_uhat, s_vhat, s_V,
             R2IJ, HIJ, DWIJ, VIJ, XIJ):
        # averaged pressure
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]
        pi = d_p[d_idx]; pj = s_p[s_idx]

        pij = rhoj * pi + rhoi * pj
        pij /= (rhoj + rhoi)

        # averaged shear viscosity
        etai = self.nu * rhoi
        etaj = self.nu * rhoj

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[0] + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[0])
        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[1] + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[1])

        # contribution due to the background pressure
        d_auhat[d_idx] += -self.pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[0]
        d_avhat[d_idx] += -self.pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[1]


class ArtificialStress(Equation):
    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_u, d_v, d_V, d_uhat, d_vhat,
             d_au, d_av, d_m, s_rho, s_u, s_v, s_V, s_uhat, s_vhat,
             DWIJ):
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]

        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]

        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # artificial stress tensor
        Axxi = rhoi*ui*(uhati - ui); Axyi = rhoi*ui*(vhati - vi)
        Ayxi = rhoi*vi*(uhati - ui); Ayyi = rhoi*vi*(vhati - vi)

        Axxj = rhoj*uj*(uhatj - uj); Axyj = rhoj*uj*(vhatj - vj)
        Ayxj = rhoj*vj*(uhatj - uj); Ayyj = rhoj*vj*(vhatj - vj)

        Ax = 0.5 * (Axxi + Axxj) * DWIJ[0] + 0.5 * (Axyi + Axyj) * DWIJ[1]
        Ay = 0.5 * (Ayxi + Ayxj) * DWIJ[0] + 0.5 * (Ayyi + Ayyj) * DWIJ[1]

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ax
        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ay
