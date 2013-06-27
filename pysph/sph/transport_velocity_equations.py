"""Functions for the transport velocity formulation of Adami et. al."""

from equations import Equation

class TransportVelocitySummationDensity(Equation):
    def __init__(self, dest, sources=None, rho0=1.0, c0=10):
        self.rho0 = rho0
        self.c0 = c0
        super(TransportVelocitySummationDensity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_V, d_rho, s_m, WIJ=1.0):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += s_m[s_idx]*WIJ

    def post_loop(self, d_idx, d_p, d_rho):
        # update the pressure using the equation of state
        d_p[d_idx] = self.c0*self.c0*( d_rho[d_idx]/self.rho0 - 1 )


class TransportVelocitySolidWall(Equation):
    def __init__(self, dest, sources=None, rho0=1.0, p0=100.0):
        self.rho0 = rho0
        self.p0 = p0
        super(TransportVelocitySolidWall, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_u, d_v, d_p, d_wij, s_u, s_v, s_p, WIJ=1.0):
        # smooth velocities at the ghost points
        d_u[d_idx] += s_u[s_idx]*WIJ
        d_v[d_idx] += s_v[s_idx]*WIJ

        # smooth pressure
        d_p[d_idx] += s_p[s_idx]*WIJ

        # denominator
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_u, d_v, d_u0, d_v0, d_p, d_rho):
        # smooth velocity at the wall particle
        if d_wij[d_idx] == 0:
            pass

        else:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]

            # solid wall condition
            d_u[d_idx] = 2*d_u0[d_idx] - d_u[d_idx]
            d_v[d_idx] = 2*d_v0[d_idx] - d_v[d_idx]

            # smooth interpolated pressure at the wall
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure
        d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + 1.0)


class TransportVelocityMomentumEquation(Equation):
    def __init__(self, dest, sources=None, pb=1.0,
                 nu=0.01, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.nu = nu
        self.pb = pb
        super(TransportVelocityMomentumEquation, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_p, d_u, d_v, d_V, d_au, d_av,
             d_auhat, d_avhat, d_m, d_uhat, d_vhat,
             s_rho, s_p, s_u, s_v, s_uhat, s_vhat, s_V,
             R2IJ=1.0, HIJ=1.0, DWIJ=[0, 0, 0], VIJ=[0, 0, 0], XIJ=[0, 0, 0]):
        # averaged pressure
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]
        pi = d_p[d_idx]; pj = s_p[s_idx]

        pij = rhoj * pi + rhoi * pj
        pij /= (rhoj + rhoi)

        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]

        # averaged shear viscosity
        etai = self.nu * rhoi
        etaj = self.nu * rhoj

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # artificial stress tensor
        Ax = 0.5 * (rhoi*ui*(uhatj - uj)*DWIJ[0] + rhoi*ui*(vhatj - vj)*DWIJ[1] + rhoj*uj*(uhati - ui)*DWIJ[0] + rhoj*uj*(vhati - vi)*DWIJ[1])
        Ay = 0.5 * (rhoi*vi*(uhatj - uj)*DWIJ[0] + rhoi*vi*(vhatj - vj)*DWIJ[1] + rhoj*vj*(uhati - ui)*DWIJ[0] + rhoj*vj*(vhati - vi)*DWIJ[1])

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[0] + Ax + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[0])

        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[1] + Ay + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[1])

        # contribution due to the background pressure
        d_auhat[d_idx] += -self.pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[0]
        d_avhat[d_idx] += -self.pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[1]
