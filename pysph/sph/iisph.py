"""The basic equations for the IISPH formulation of

    M. Ihmsen, J. Cornelis, B. Solenthaler, C. Horvath, M. Teschner, "Implicit
    Incompressible SPH," IEEE Transactions on Visualization and Computer
    Graphics, vol. 20, no. 3, pp. 426-435, March 2014.
    http://dx.doi.org/10.1109/TVCG.2013.105

"""

from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep


class IISPHStep(IntegratorStep):
    """A straightforward and simple integrator to be used for IISPH.
    """
    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_uadv, d_vadv, d_wadv, d_au, d_av, d_aw,
               d_ax, d_ay, d_az, dt=0.0):
        d_u[d_idx] = d_uadv[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_vadv[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_wadv[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] += dt * d_ax[d_idx]
        d_y[d_idx] += dt * d_ay[d_idx]
        d_z[d_idx] += dt * d_az[d_idx]


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


class AdvectionAcceleration(Equation):
    def __init__(self, dest, sources=None, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(AdvectionAcceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_uadv, d_vadv, d_wadv):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz
        d_uadv[d_idx] = 0.0
        d_vadv[d_idx] = 0.0
        d_wadv[d_idx] = 0.0

    def loop(self):
        pass

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_uadv, d_vadv, d_wadv,
                  d_u, d_v, d_w, dt=0.0):
        d_uadv[d_idx] = d_u[d_idx] + dt*d_au[d_idx]
        d_vadv[d_idx] = d_v[d_idx] + dt*d_av[d_idx]
        d_wadv[d_idx] = d_w[d_idx] + dt*d_aw[d_idx]


class ComputeDII(Equation):
    def initialize(self, d_idx, d_dii0, d_dii1, d_dii2):
        d_dii0[d_idx] = 0.0
        d_dii1[d_idx] = 0.0
        d_dii2[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_dii0, d_dii1, d_dii2,
             s_idx, s_m, DWIJ, dt=0.0):
        rho_1 = 1.0/d_rho[d_idx]
        fac = -dt*dt*s_m[s_idx]*rho_1*rho_1
        d_dii0[d_idx] += fac*DWIJ[0]
        d_dii1[d_idx] += fac*DWIJ[1]
        d_dii2[d_idx] += fac*DWIJ[2]


class ComputeRhoAdvection(Equation):
    def initialize(self, d_idx, d_rho_adv, d_rho, d_p0, d_p, d_aii):
        d_rho_adv[d_idx] = d_rho[d_idx]
        d_p0[d_idx] = 0.5*d_p[d_idx]

    def loop(self, d_idx, d_rho_adv, d_uadv, d_vadv, d_wadv, d_u, d_v, d_w,
             s_idx, s_m, s_uadv, s_vadv, s_wadv, s_u, DWIJ, dt=0.0):

        vijdotdwij = (d_uadv[d_idx] - s_uadv[s_idx])*DWIJ[0] + \
                     (d_vadv[d_idx] - s_vadv[s_idx])*DWIJ[1] + \
                     (d_wadv[d_idx] - s_wadv[s_idx])*DWIJ[2]

        d_rho_adv[d_idx] += dt*s_m[s_idx]*vijdotdwij


class ComputeAII(Equation):
    def initialize(self, d_idx, d_aii):
        d_aii[d_idx] = 0.0

    def loop(self, d_idx, d_aii, d_dii0, d_dii1, d_dii2, d_m, d_rho,
             s_idx, s_m, s_rho, DWIJ, dt=0.0):
        rho1 = 1.0/d_rho[d_idx]
        fac = dt*dt*d_m[d_idx]*rho1*rho1
        # The following is m_j (d_ii - d_ji) . DWIJ
        # DWIJ = -DWJI
        dijdotdwij = (d_dii0[d_idx] - fac*DWIJ[0])*DWIJ[0] + \
                     (d_dii1[d_idx] - fac*DWIJ[1])*DWIJ[1] + \
                     (d_dii2[d_idx] - fac*DWIJ[2])*DWIJ[2]
        d_aii[d_idx] += s_m[s_idx]*dijdotdwij


class ComputeDIJPJ(Equation):
    def initialize(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2):
        d_dijpj0[d_idx] = 0.0
        d_dijpj1[d_idx] = 0.0
        d_dijpj2[d_idx] = 0.0

    def loop(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_m, s_rho, s_p0, DWIJ, dt=0.0):
        rho1 = 1.0/s_rho[s_idx]
        fac = -dt*dt*s_m[s_idx]*rho1*rho1*s_p0[s_idx]
        d_dijpj0[d_idx] += fac*DWIJ[0]
        d_dijpj1[d_idx] += fac*DWIJ[1]
        d_dijpj2[d_idx] += fac*DWIJ[2]


class PressureSolve(Equation):
    def __init__(self, dest, sources=None, rho0=1.0, omega=0.5,
                 tolerance=1e-2, debug=False):
        self.rho0 = rho0
        self.omega = omega
        self.compression = 0.0
        self.count = 0
        self.debug = debug
        self.tolerance = tolerance
        super(PressureSolve, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0
        self.compression = 0.0
        self.count = 0

    def loop(self, d_idx, d_p, d_p0, d_rho, d_m, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_m, s_dii0, s_dii1, s_dii2,
             s_p0, s_dijpj0, s_dijpj1, s_dijpj2, DWIJ, dt=0.0):

        # Note that a good way to check this is to see that when
        # d_idx == s_idx the contribution is zero, as is expected.
        rho1 = 1.0/d_rho[d_idx]
        fac = dt*dt*d_m[d_idx]*rho1*rho1*d_p0[d_idx]
        djkpk0 = s_dijpj0[s_idx] - fac*DWIJ[0]
        djkpk1 = s_dijpj1[s_idx] - fac*DWIJ[1]
        djkpk2 = s_dijpj2[s_idx] - fac*DWIJ[2]

        tmp0 = d_dijpj0[d_idx] - s_dii0[s_idx]*s_p0[s_idx] - djkpk0
        tmp1 = d_dijpj1[d_idx] - s_dii1[s_idx]*s_p0[s_idx] - djkpk1
        tmp2 = d_dijpj2[d_idx] - s_dii2[s_idx]*s_p0[s_idx] - djkpk2
        tmpdotdwij = tmp0*DWIJ[0] + tmp1*DWIJ[1] + tmp2*DWIJ[2]

        # This is corrected in the post_loop.
        d_p[d_idx] += s_m[s_idx]*tmpdotdwij

    def post_loop(self, d_idx, d_p0, d_p, d_aii, d_rho_adv, d_rho):

        tmp = self.rho0 - d_rho_adv[d_idx] - d_p[d_idx]
        p = (1.0 - self.omega)*d_p0[d_idx] + self.omega/d_aii[d_idx]*tmp

        # Clamp pressure to positive values.
        if p < 0.0:
            p = 0.0
        else:
            compression =  p*d_aii[d_idx] - tmp
            self.compression += compression
            self.count += 1

        d_p0[d_idx] = p
        d_p[d_idx] = p

    def converged(self):
        count = self.count
        debug = self.debug
        ratio = abs(self.compression/count/self.rho0) if count > 0 else 0

        if ratio > self.tolerance:
            if debug:
                print "Not converged:", ratio
            return -1.0
        else:
            if debug:
                print "Converged:", ratio
            return 1.0


class PressureForce(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_p, d_au, d_av, d_aw,
             s_idx, s_m, s_rho, s_p, DWIJ):
        rhoi1 = 1.0/d_rho[d_idx]
        rhoj1 = 1.0/s_rho[s_idx]
        fac = -s_m[s_idx]*(d_p[d_idx]*rhoi1*rhoi1 + s_p[s_idx]*rhoj1*rhoj1)
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]
