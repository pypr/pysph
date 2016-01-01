"""The basic equations for the IISPH formulation of

    M. Ihmsen, J. Cornelis, B. Solenthaler, C. Horvath, M. Teschner, "Implicit
    Incompressible SPH," IEEE Transactions on Visualization and Computer
    Graphics, vol. 20, no. 3, pp. 426-435, March 2014.
    http://dx.doi.org/10.1109/TVCG.2013.105

"""

from numpy import sqrt
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array


class IISPHStep(IntegratorStep):
    """A straightforward and simple integrator to be used for IISPH.
    """
    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_uadv, d_vadv, d_wadv, d_au, d_av, d_aw,
               d_ax, d_ay, d_az, dt):
        d_u[d_idx] = d_uadv[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_vadv[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_wadv[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]


class NumberDensity(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ

class SummationDensityBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(SummationDensityBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, s_idx, s_V, WIJ):
        d_rho[d_idx] += self.rho0/s_V[s_idx]*WIJ

class NormalizedSummationDensity(Equation):
    def initialize(self, d_idx, d_rho, d_rho_adv, d_rho0, d_V):
        d_rho0[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0
        d_rho_adv[d_idx] = 0.0
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_rho_adv, d_V, s_idx, s_m, s_rho0, WIJ):
        tmp = s_m[s_idx]*WIJ
        d_rho[d_idx] += tmp
        d_rho_adv[d_idx] += tmp/s_rho0[s_idx]
        d_V[d_idx] += WIJ

    def post_loop(self, d_idx, d_rho, d_rho_adv):
        d_rho[d_idx] /= d_rho_adv[d_idx]


class AdvectionAcceleration(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
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

class ViscosityAcceleration(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(ViscosityAcceleration, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw, s_idx, s_m, EPS,
             VIJ, XIJ, RHOIJ1, R2IJ, DWIJ):
        dwijdotxij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        fac = 2.0*self.nu*s_m[s_idx]*RHOIJ1*dwijdotxij/(R2IJ + EPS)
        d_au[d_idx] += fac*VIJ[0]
        d_av[d_idx] += fac*VIJ[1]
        d_aw[d_idx] += fac*VIJ[2]

class ViscosityAccelerationBoundary(Equation):
    """The acceleration on the fluid due to a boundary.
    """
    def __init__(self, dest, sources, rho0, nu):
        self.nu = nu
        self.rho0 = rho0
        super(ViscosityAccelerationBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw, d_rho, s_idx, s_V, EPS,
             VIJ, XIJ, R2IJ, DWIJ):
        phi_b = self.rho0/(s_V[s_idx]*d_rho[d_idx])
        dwijdotxij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        fac = 2.0*self.nu*phi_b*dwijdotxij/(R2IJ + EPS)
        d_au[d_idx] += fac*VIJ[0]
        d_av[d_idx] += fac*VIJ[1]
        d_aw[d_idx] += fac*VIJ[2]


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


class ComputeDIIBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeDIIBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_dii0, d_dii1, d_dii2, d_rho,
             s_idx, s_m, s_V, DWIJ, dt=0.0):
        rhoi1 = 1.0/d_rho[d_idx]
        fac = -dt*dt*rhoi1*rhoi1*self.rho0/s_V[s_idx]
        d_dii0[d_idx] += fac*DWIJ[0]
        d_dii1[d_idx] += fac*DWIJ[1]
        d_dii2[d_idx] += fac*DWIJ[2]


class ComputeRhoAdvection(Equation):
    def initialize(self, d_idx, d_rho_adv, d_rho, d_p0, d_p, d_piter, d_aii):
        d_rho_adv[d_idx] = d_rho[d_idx]
        d_p0[d_idx] = d_p[d_idx]
        d_piter[d_idx] = 0.5*d_p[d_idx]

    def loop(self, d_idx, d_rho, d_rho_adv, d_uadv, d_vadv, d_wadv, d_u, d_v, d_w,
             s_idx, s_m, s_uadv, s_vadv, s_wadv, DWIJ, dt=0.0):

        vijdotdwij = (d_uadv[d_idx] - s_uadv[s_idx])*DWIJ[0] + \
                     (d_vadv[d_idx] - s_vadv[s_idx])*DWIJ[1] + \
                     (d_wadv[d_idx] - s_wadv[s_idx])*DWIJ[2]

        d_rho_adv[d_idx] += dt*s_m[s_idx]*vijdotdwij


class ComputeRhoBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeRhoBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_rho_adv, d_uadv, d_vadv, d_wadv,
             s_idx, s_u, s_v, s_w, s_V, WIJ, DWIJ, dt=0.0):
        phi_b = self.rho0/s_V[s_idx]

        vijdotdwij = (d_uadv[d_idx] - s_u[s_idx])*DWIJ[0] + \
                     (d_vadv[d_idx] - s_v[s_idx])*DWIJ[1] + \
                     (d_wadv[d_idx] - s_w[s_idx])*DWIJ[2]
        d_rho_adv[d_idx] += dt*phi_b*vijdotdwij


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


class ComputeAIIBoundary(Equation):
    """ This is important and not really discussed in the original IISPH paper.
    """
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(ComputeAIIBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_aii, d_dii0, d_dii1, d_dii2, d_rho,
             s_idx, s_m, s_V, DWIJ, dt=0.0):
        phi_b = self.rho0/s_V[s_idx]
        dijdotdwij = d_dii0[d_idx]*DWIJ[0] + d_dii1[d_idx]*DWIJ[1] + \
                     d_dii2[d_idx]*DWIJ[2]
        d_aii[d_idx] += phi_b*dijdotdwij


class ComputeDIJPJ(Equation):
    def initialize(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2):
        d_dijpj0[d_idx] = 0.0
        d_dijpj1[d_idx] = 0.0
        d_dijpj2[d_idx] = 0.0

    def loop(self, d_idx, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_m, s_rho, s_piter, DWIJ, dt=0.0):
        rho1 = 1.0/s_rho[s_idx]
        fac = -dt*dt*s_m[s_idx]*rho1*rho1*s_piter[s_idx]
        d_dijpj0[d_idx] += fac*DWIJ[0]
        d_dijpj1[d_idx] += fac*DWIJ[1]
        d_dijpj2[d_idx] += fac*DWIJ[2]


class PressureSolve(Equation):
    def __init__(self, dest, sources, rho0, omega=0.5,
                 tolerance=1e-2, debug=False):
        self.rho0 = rho0
        self.omega = omega
        self.compression = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(PressureSolve, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_compression):
        d_p[d_idx] = 0.0
        d_compression[d_idx] = 0.0

    def loop(self, d_idx, d_p, d_piter, d_rho, d_m, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_m, s_dii0, s_dii1, s_dii2,
             s_piter, s_dijpj0, s_dijpj1, s_dijpj2, DWIJ, dt=0.0):

        # Note that a good way to check this is to see that when
        # d_idx == s_idx the contribution is zero, as is expected.
        rho1 = 1.0/d_rho[d_idx]
        fac = dt*dt*d_m[d_idx]*rho1*rho1*d_piter[d_idx]
        djkpk0 = s_dijpj0[s_idx] - fac*DWIJ[0]
        djkpk1 = s_dijpj1[s_idx] - fac*DWIJ[1]
        djkpk2 = s_dijpj2[s_idx] - fac*DWIJ[2]

        tmp0 = d_dijpj0[d_idx] - s_dii0[s_idx]*s_piter[s_idx] - djkpk0
        tmp1 = d_dijpj1[d_idx] - s_dii1[s_idx]*s_piter[s_idx] - djkpk1
        tmp2 = d_dijpj2[d_idx] - s_dii2[s_idx]*s_piter[s_idx] - djkpk2
        tmpdotdwij = tmp0*DWIJ[0] + tmp1*DWIJ[1] + tmp2*DWIJ[2]

        # This is corrected in the post_loop.
        d_p[d_idx] += s_m[s_idx]*tmpdotdwij

    def post_loop(self, d_idx, d_piter, d_p0, d_p, d_aii, d_rho_adv, d_rho,
                  d_compression, dt=0.0):
        #tmp = d_rho[d_idx] - d_rho_adv[d_idx] - d_p[d_idx]
        tmp = self.rho0 - d_rho_adv[d_idx] - d_p[d_idx]
        p = (1.0 - self.omega)*d_piter[d_idx] + self.omega/d_aii[d_idx]*tmp

        aii_min = dt*dt*0.01

        # Clamp pressure to positive values.
        if p < 0.0:
            p = 0.0
        elif abs(d_aii[d_idx]) < aii_min :
            p = 0.0
        else:
            d_compression[d_idx] = abs(p*d_aii[d_idx] - tmp)

        d_piter[d_idx] = p
        d_p[d_idx] = p

    def reduce(self, dst):
        dst.tmp_comp[0] = serial_reduce_array(dst.array.compression > 0.0, 'sum')
        dst.tmp_comp[1] = serial_reduce_array(dst.array.compression, 'sum')
        dst.tmp_comp.set_data(parallel_reduce_array(dst.tmp_comp, 'sum'))
        if dst.tmp_comp[0] > 0:
            comp = dst.tmp_comp[1]/dst.tmp_comp[0]/self.rho0
        else:
            comp = 0.0
        self.compression = comp

    def converged(self):
        debug = self.debug
        compression = self.compression

        if compression > self.tolerance:
            if debug:
                print("Not converged:", compression)
            return -1.0
        else:
            if debug:
                print("Converged:", compression)
            return 1.0

class PressureSolveBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureSolveBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho, d_dijpj0, d_dijpj1, d_dijpj2,
             s_idx, s_V, DWIJ):
        phi_b = self.rho0/s_V[s_idx]
        dijdotwij = d_dijpj0[d_idx]*DWIJ[0] + \
                    d_dijpj1[d_idx]*DWIJ[1] + \
                    d_dijpj2[d_idx]*DWIJ[2]
        d_p[d_idx] += phi_b*dijdotwij


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

    def post_loop(self, d_idx, d_au, d_av, d_aw,
                  d_uadv, d_vadv, d_wadv, DT_ADAPT):
        fac = d_au[d_idx]*d_au[d_idx] + d_av[d_idx]*d_av[d_idx] +\
                   d_aw[d_idx]*d_aw[d_idx]
        vmag = sqrt(d_uadv[d_idx]*d_uadv[d_idx] + d_vadv[d_idx]*d_vadv[d_idx] +
                    d_wadv[d_idx]*d_wadv[d_idx])
        DT_ADAPT[0] = max(2.0*vmag, DT_ADAPT[0])
        DT_ADAPT[1] = max(2.0*fac, DT_ADAPT[1])


class PressureForceBoundary(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureForceBoundary, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_au, d_av, d_aw,  d_p, s_idx, s_V, DWIJ):
        rho1 = 1.0/d_rho[d_idx]
        fac = -d_p[d_idx]*rho1*rho1*self.rho0/s_V[s_idx]
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]
