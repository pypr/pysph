from math import exp
from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.sph.gas_dynamics.riemann_solver import (HELPERS, riemann_solve,
                                                   printf)
from pysph.base.particle_array import get_ghost_tag

# Constants
GHOST_TAG = get_ghost_tag()

# Riemann solver types
NonDiffusive = 0
VanLeer = 1
Exact = 2
HLLC = 3
Ducowicz = 4
HLLE = 5
Roe = 6
LLXF = 7
HLLCBall = 8
HLLBall = 9
HLLSY = 10

# GSPHInterpolationType
Delta = 0
Linear = 1
Cubic = 2


def sgn(x=0.0):
    return (x > 0) - (x < 0)


def monotonicity_min(_x1=0.0, _x2=0.0, _x3=0.0):
    x1, x2, x3, _min = declare('double', 4)
    x1 = 2.0 * abs(_x1)
    x2 = abs(_x2)
    x3 = 2.0 * abs(_x3)

    sx1 = sgn(_x1)
    sx2 = sgn(_x2)
    sx3 = sgn(_x3)

    if ((sx1 != sx2) or (sx2 != sx3)):
        return 0.0
    else:
        if x2 < x1:
            if x3 < x2:
                _min = x3
            else:
                _min = x2
        else:
            if x3 < x1:
                _min = x3
            else:
                _min = x1

    return sx1 * _min


class GSPHGradients(Equation):
    def initialize(self, d_idx, d_px, d_py, d_pz, d_ux, d_uy, d_uz,
                   d_vx, d_vy, d_vz, d_wx, d_wy, d_wz):
        d_px[d_idx] = 0.0
        d_py[d_idx] = 0.0
        d_pz[d_idx] = 0.0
        d_ux[d_idx] = 0.0
        d_uy[d_idx] = 0.0
        d_uz[d_idx] = 0.0
        d_vx[d_idx] = 0.0
        d_vy[d_idx] = 0.0
        d_vz[d_idx] = 0.0
        d_wx[d_idx] = 0.0
        d_wy[d_idx] = 0.0
        d_wz[d_idx] = 0.0

    def loop(self, d_idx, d_px, d_py, d_pz, d_ux, d_uy, d_uz,
             d_vx, d_vy, d_vz, d_wx, d_wy, d_wz, d_p, d_u, d_v, d_w,
             s_idx, s_p, s_u, s_v, s_w, s_rho, s_m,
             DWI):
        rj1 = 1.0/s_rho[s_idx]
        pji = s_p[s_idx] - d_p[d_idx]
        uji = s_u[s_idx] - d_u[d_idx]
        vji = s_v[s_idx] - d_v[d_idx]
        wji = s_w[s_idx] - d_w[d_idx]
        tmp = rj1*s_m[s_idx]*pji
        d_px[d_idx] += tmp*DWI[0]
        d_py[d_idx] += tmp*DWI[1]
        d_pz[d_idx] += tmp*DWI[2]
        tmp = rj1*s_m[s_idx]*uji
        d_ux[d_idx] += tmp*DWI[0]
        d_uy[d_idx] += tmp*DWI[1]
        d_uz[d_idx] += tmp*DWI[2]
        tmp = rj1*s_m[s_idx]*vji
        d_vx[d_idx] += tmp*DWI[0]
        d_vy[d_idx] += tmp*DWI[1]
        d_vz[d_idx] += tmp*DWI[2]
        tmp = rj1*s_m[s_idx]*wji
        d_wx[d_idx] += tmp*DWI[0]
        d_wy[d_idx] += tmp*DWI[1]
        d_wz[d_idx] += tmp*DWI[2]


class GSPHUpdateGhostProps(Equation):
    """Copy the GSPH gradients and other props required for GSPH
    from real particle to ghost particles

    """
    def __init__(self, dest, sources=None):
        super(GSPHUpdateGhostProps, self).__init__(dest, sources)
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_tag, d_orig_idx, d_px, d_py, d_pz,
                   d_ux, d_uy, d_uz, d_vx, d_vy, d_vz, d_wx, d_wy, d_wz,
                   d_grhox, d_grhoy, d_grhoz, d_dwdh, d_rho, d_div,
                   d_p, d_cs):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            # copy pressure grads
            d_px[d_idx] = d_px[idx]
            d_py[d_idx] = d_py[idx]
            d_pz[d_idx] = d_pz[idx]
            # copy u grads
            d_ux[d_idx] = d_ux[idx]
            d_uy[d_idx] = d_uy[idx]
            d_uz[d_idx] = d_uz[idx]
            # copy u grads
            d_vx[d_idx] = d_vx[idx]
            d_vy[d_idx] = d_vy[idx]
            d_vz[d_idx] = d_vz[idx]
            # copy u grads
            d_wx[d_idx] = d_wx[idx]
            d_wy[d_idx] = d_wy[idx]
            d_wz[d_idx] = d_wz[idx]
            # copy density grads
            d_grhox[d_idx] = d_grhox[idx]
            d_grhoy[d_idx] = d_grhoy[idx]
            d_grhoz[d_idx] = d_grhoz[idx]
            # other misc props
            d_dwdh[d_idx] = d_dwdh[idx]
            d_rho[d_idx] = d_rho[idx]
            d_div[d_idx] = d_div[idx]
            d_p[d_idx] = d_p[idx]
            d_cs[d_idx] = d_cs[idx]


class GSPHAcceleration(Equation):
    """Class to implement the GSPH acclerations.

    We implement Inutsuka's original GSPH algorithm I02 defined in
    'Reformulation of Smoothed Particle Hydrodynamics with Riemann
    Solver', (2002), JCP, 179, 238--267 and Iwasaki and Inutsuka's
    vesion (specifically the monotonicity constraint) described in
    'Smoothed particle magnetohydrodynamics with a Riemann solver and
    the method of characteristics' 2011, MNRAS, referred to as IwIn

    Additional details about the algorithm are also described by
    Murante et al.

    """
    def __init__(self, dest, sources, g1=0.0, g2=0.0,
                 monotonicity=0, rsolver=Exact,
                 interpolation=Linear, interface_zero=True, hybrid=False,
                 blend_alpha=5.0, tf=1.0,
                 gamma=1.4, niter=20, tol=1e-6):
        """
        Parameters
        ----------
        g1, g2 : double
            ADKE style thermal conduction parameters
        rsolver: int
            Riemann solver to use.  See pysph.sph.gas_dynamics.gsph for
            valid options.
        interpolation: int
            Kind of interpolation for the specific volume integrals.
        monotonicity : int
            Type of monotonicity algorithm to use:
            0 : First order GSPH
            1 : I02 algorithm
            2 : IwIn algorithm
        interface_zero : bool
            Set Interface position s^*_{ij} = 0 for the Riemann problem.
        hybrid, blend_alpha : bool, double
            Hybrid scheme and blending alpha value
        tf: float
            Final time of simulation for using in blending.
        gamma: float
            Gamma for Equation of state.
        niter: int
            Max number of iterations for iterative Riemann solvers.
        tol: double
            Tolerance for iterative Riemann solvers.
        """
        self.gamma = gamma
        self.niter = niter
        self.tol = tol
        self.g1 = g1
        self.g2 = g2
        self.monotonicity = monotonicity
        self.interpolation = interpolation
        self.rsolver = rsolver
        # Interface position for data reconstruction
        self.sstar = 0.0
        if (g1 == 0 and g2 == 0):
            self.thermal_conduction = 0
        else:
            self.thermal_conduction = 1
        self.interface_zero = interface_zero
        self.hybrid = hybrid
        self.blend_alpha = blend_alpha
        self.tf = tf

        super(GSPHAcceleration, self).__init__(dest, sources)

    def _get_helpers_(self):
        return HELPERS + [monotonicity_min, sgn]

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_h, d_rho, d_cs, d_div, d_p, d_e, d_grhox,
             d_grhoy, d_grhoz, d_u, d_v, d_w, d_px, d_py, d_pz, d_ux, d_uy,
             d_uz, d_vx, d_vy, d_vz, d_wx, d_wy, d_wz, d_au, d_av, d_aw, d_ae,
             s_idx, s_rho, s_m, s_h, s_cs, s_div, s_p, s_e, s_grhox,
             s_grhoy, s_grhoz, s_u, s_v, s_w, s_px, s_py, s_pz,
             s_ux, s_uy, s_uz, s_vx, s_vy, s_vz, s_wx, s_wy, s_wz,
             XIJ, DWIJ, DWI, DWJ, RIJ, RHOIJ, EPS, dt, t):
        blending_factor = exp(-self.blend_alpha*t/self.tf)
        g1 = self.g1
        g2 = self.g2
        hi = d_h[d_idx]
        hj = s_h[s_idx]
        eij = declare('matrix(3)')
        if RIJ < 1e-14:
            eij[0] = 0.0
            eij[1] = 0.0
            eij[2] = 0.0
            sij = 1.0/(RIJ + EPS)
        else:
            eij[0] = XIJ[0]/RIJ
            eij[1] = XIJ[1]/RIJ
            eij[2] = XIJ[2]/RIJ
            sij = 1.0/RIJ

        vl = s_u[s_idx]*eij[0] + s_v[s_idx]*eij[1] + s_w[s_idx]*eij[2]
        vr = d_u[d_idx]*eij[0] + d_v[d_idx]*eij[1] + d_w[d_idx]*eij[2]

        Hi = g1*hi*d_cs[d_idx] + g2*hi*hi*(abs(d_div[d_idx]) - d_div[d_idx])

        grhoi_dot_eij = (d_grhox[d_idx]*eij[0] + d_grhoy[d_idx]*eij[1]
                         + d_grhoz[d_idx]*eij[2])
        grhoj_dot_eij = (s_grhox[s_idx]*eij[0] + s_grhoy[s_idx]*eij[1]
                         + s_grhoz[s_idx]*eij[2])

        sp_vol = declare('matrix(3)')
        self.interpolate(hi, hj, d_rho[d_idx], s_rho[s_idx], RIJ,
                         grhoi_dot_eij, grhoj_dot_eij, sp_vol)
        vij_i = sp_vol[0]
        vij_j = sp_vol[1]
        sstar = sp_vol[2]
        # Gradients in the local coordinate system
        rsi = (d_grhox[d_idx]*eij[0] + d_grhoy[d_idx]*eij[1] +
               d_grhoz[d_idx]*eij[2])
        psi = (d_px[d_idx]*eij[0] + d_py[d_idx]*eij[1] + d_pz[d_idx]*eij[2])
        vsi = (eij[0]*eij[0]*d_ux[d_idx] +
               eij[0]*eij[1]*(d_uy[d_idx] + d_vx[d_idx]) +
               eij[0]*eij[2]*(d_uz[d_idx] + d_wx[d_idx]) +
               eij[1]*eij[1]*d_vy[d_idx] +
               eij[1]*eij[2]*(d_vz[d_idx] + d_wy[d_idx]) +
               eij[2]*eij[2]*d_wz[d_idx])

        rsj = (s_grhox[s_idx]*eij[0] + s_grhoy[s_idx]*eij[1] +
               s_grhoz[s_idx]*eij[2])
        psj = (s_px[s_idx]*eij[0] + s_py[s_idx]*eij[1] + s_pz[s_idx]*eij[2])
        vsj = (eij[0]*eij[0]*s_ux[s_idx] +
               eij[0]*eij[1]*(s_uy[s_idx] + s_vx[s_idx]) +
               eij[0]*eij[2]*(s_uz[s_idx] + s_wx[s_idx]) +
               eij[1]*eij[1]*s_vy[s_idx] +
               eij[1]*eij[2]*(s_vz[s_idx] + s_wy[s_idx]) +
               eij[2]*eij[2]*s_wz[s_idx])

        csi = d_cs[d_idx]
        csj = s_cs[s_idx]
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        if self.monotonicity == 0:  # First order scheme
            rsi = 0.0
            rsj = 0.0
            psi = 0.0
            psj = 0.0
            vsi = 0.0
            vsj = 0.0

        if self.monotonicity == 1:  # I02 algorithm
            if (vsi * vsj) < 0:
                vsi = 0.
                vsj = 0.

            # default to first order near a shock
            if (min(csi, csj) < 3.0*(vl - vr)):
                rsi = 0.
                rsj = 0.
                psi = 0.
                psj = 0.
                vsi = 0.
                vsj = 0.

        if self.monotonicity == 2 and RIJ > 1e-14:  # IwIn algorithm
            qijr = rhoi - rhoj
            qijp = pi - pj
            qiju = vr - vl

            delr = rsi * RIJ
            delrp = 2 * delr - qijr
            delp = psi * RIJ
            delpp = 2 * delp - qijp
            delv = vsi * RIJ
            delvp = 2 * delv - qiju

            # corrected values for i
            rsi = monotonicity_min(qijr, delr, delrp)/RIJ
            psi = monotonicity_min(qijp, delp, delpp)/RIJ
            vsi = monotonicity_min(qiju, delv, delvp)/RIJ

            delr = rsj * RIJ
            delrp = 2 * delr - qijr
            delp = psj * RIJ
            delpp = 2 * delp - qijp
            delv = vsj * RIJ
            delvp = 2 * delv - qiju

            # corrected values for j
            rsj = monotonicity_min(qijr, delr, delrp)/RIJ
            psj = monotonicity_min(qijp, delp, delpp)/RIJ
            vsj = monotonicity_min(qiju, delv, delvp)/RIJ
        elif self.monotonicity == 2 and RIJ < 1e-14:  # IwIn algorithm
            rsi = 0.0
            rsj = 0.0
            psi = 0.0
            psj = 0.0
            vsi = 0.0
            vsj = 0.0

        # Input to the riemann solver
        sstar *= 2.0

        # left and right density
        rhol = rhoj + 0.5 * rsj * RIJ * (1.0 - csj*dt*sij + sstar)
        rhor = rhoi - 0.5 * rsi * RIJ * (1.0 - csi*dt*sij + sstar)

        # corrected density
        if rhol < 0:
            rhol = rhoj
        if rhor < 0:
            rhor = rhoi

        # left and right pressure
        pl = pj + 0.5 * psj * RIJ * (1.0 - csj*dt*sij + sstar)
        pr = pi - 0.5 * psi * RIJ * (1.0 - csi*dt*sij + sstar)

        # corrected pressure
        if pl < 0:
            pl = pj
        if pr < 0:
            pr = pi

        # left and right velocity
        ul = vl + 0.5 * vsj * RIJ * (1.0 - csj*dt*sij + sstar)
        ur = vr - 0.5 * vsi * RIJ * (1.0 - csi*dt*sij + sstar)

        # Intermediate state from the Riemann solver
        result = declare('matrix(2)')
        riemann_solve(
            self.rsolver, rhol, rhor, pl, pr, ul, ur,
            self.gamma, self.niter, self.tol, result
        )
        pstar = result[0]
        ustar = result[1]

        # blend of two intermediate states
        if self.hybrid:
            riemann_solve(
                10, rhoj, rhoi, pl, pr, vl, vr, self.gamma,
                self.niter, self.tol, result
            )
            pstar2 = result[0]
            ustar2 = result[1]
            ustar = ustar + blending_factor * (ustar2 - ustar)
            pstar = pstar + blending_factor * (pstar2 - pstar)

        # three dimensional velocity (70)
        vstar = declare('matrix(3)')
        vstar[0] = ustar*eij[0]
        vstar[1] = ustar*eij[1]
        vstar[2] = ustar*eij[2]

        # velocity accelerations
        mj = s_m[s_idx]
        d_au[d_idx] += -mj * pstar * (vij_i * DWI[0] + vij_j * DWJ[0])
        d_av[d_idx] += -mj * pstar * (vij_i * DWI[1] + vij_j * DWJ[1])
        d_aw[d_idx] += -mj * pstar * (vij_i * DWI[2] + vij_j * DWJ[2])

        # contribution to the thermal energy term
        # (85). The contribution due to \dot{x}^*_i will
        # be added in the integrator.
        vstardotdwi = vstar[0]*DWI[0] + vstar[1]*DWI[1] + vstar[2]*DWI[2]
        vstardotdwj = vstar[0]*DWJ[0] + vstar[1]*DWJ[1] + vstar[2]*DWJ[2]
        d_ae[d_idx] += -mj * pstar * (vij_i * vstardotdwi +
                                      vij_j * vstardotdwj)

        # artificial thermal conduction terms
        if self.thermal_conduction:
            divj = s_div[s_idx]
            Hj = g1 * hj * csj + g2 * hj*hj * (abs(divj) - divj)
            Hij = (Hi + Hj) * (d_e[d_idx] - s_e[s_idx])

            Hij /= (RHOIJ * (RIJ*RIJ + EPS))

            d_ae[d_idx] += mj*Hij*(XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] +
                                   XIJ[2]*DWIJ[2])

    def interpolate(self, hi=0.0, hj=0.0, rhoi=0.0, rhoj=0.0,
                    sij=0.0, gri_eij=0.0, grj_eij=0.0,
                    result=[0.0, 0.0, 0.0]):
        """Interpolation for the specific volume integrals in GSPH.

        Parameters:
        -----------

        hi, hj : double
            Particle smoothing length (scale?) at i (right) and j (left)

        rhoi, rhoj : double
            Particle densities at i (right) and j (left)

        sij : double
            Particle separation in the local coordinate system (si - sj)

        gri_eij, grj_eij : double
            Gradient of density at the particles in the global coordinate
            system dot product with eij

        Notes:
        ------

        The interpolation scheme determines the form of the 'specific
        volume' contributions Vij^2 in the GSPH equations.

        The simplest Delta or point interpolation uses Vi^2 =
        1./rho_i^2.

        The more involved linear or cubic spline interpolations are
        defined in the GSPH references.

        Most recent papers on GSPH typically assume the interface to
        be located midway between the particles. In the local
        coordinate system, this corresponds to sij = 0. From I02, it
        seems the definition is only really valid for the constant
        smoothing length case. Set interface_zero to False if you want
        to include this term

        """
        Vi = 1./rhoi
        Vj = 1./rhoj

        Vip = -1./(rhoi*rhoi) * gri_eij
        Vjp = -1./(rhoj*rhoj) * grj_eij

        aij, bij, cij, dij, hij, vij, vij_i, vij_j = declare('double', 8)
        hij = 0.5 * (hi + hj)
        sstar = self.sstar

        # simplest delta or point interpolation
        if self.interpolation == 0:
            vij_i2 = 1./(rhoi * rhoi)
            vij_j2 = 1./(rhoj * rhoj)

        # linear interpolation
        elif self.interpolation == 1:
            # avoid singularities
            if sij < 1e-8:
                cij = 0.0
            else:
                cij = (Vi - Vj)/sij

            dij = 0.5 * (Vi + Vj)

            vij_i2 = 0.25 * hi * hi * cij * cij + dij * dij
            vij_j2 = 0.25 * hj * hj * cij * cij + dij * dij

            # approximate value for the interface location when using
            # variable smoothing lengths
            if not self.interface_zero:
                vij = 0.5 * (vij_i2 + vij_j2)
                sstar = 0.5 * hij*hij * cij*dij/vij

        # cubic spline interpolation
        elif self.interpolation == 2:
            if sij < 1e-8:
                aij = bij = cij = 0.0
                dij = 0.5 * (Vi + Vj)
            else:
                aij = -2.0 * (Vi - Vj)/(sij*sij*sij) + (Vip + Vjp)/(sij*sij)
                bij = 0.5 * (Vip - Vjp)/sij
                cij = 1.5 * (Vi - Vj)/sij - 0.25 * (Vip + Vjp)
                dij = 0.5 * (Vi + Vj) - 0.125*(Vip - Vjp)*sij

            hi2 = hi*hi
            hj2 = hj*hj
            hi4 = hi2*hi2
            hj4 = hj2*hj2
            hi6 = hi4*hi2
            hj6 = hj4*hj2

            vij_i2 = ((15.0)/(64.0)*hi6 * aij*aij +
                      (3.0)/(16.0) * hi4 * (2*aij*cij + bij*bij) +
                      0.25*hi2*(2*bij*dij + cij*cij) + dij * dij)

            vij_j2 = ((15.0)/(64.0)*hj6 * aij*aij +
                      (3.0)/(16.0) * hj4 * (2*aij*cij + bij*bij) +
                      0.25*hj2 * (2*bij*dij + cij*cij) + dij * dij)
            hij2 = hij*hij
            hij4 = hij2*hij2
            if not self.interface_zero:
                vij = 0.5*(vij_i2 + vij_j2)
                sstar = ((15.0/32.0)*hij4*hij2*aij*bij +
                         (3.0/8.0)*hij4*(aij*dij + bij*cij) +
                         0.5*hij2*cij*dij)/vij
        else:
            printf("%s", "Unknown interpolation type")

        result[0] = vij_i2
        result[1] = vij_j2
        result[2] = sstar
