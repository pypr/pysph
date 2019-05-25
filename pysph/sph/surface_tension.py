"""
Implementation of the equations used for surface tension modelling,
for example in KHI simulations. The references are as under:

 - M. Shadloo, M. Yildiz, "Numerical modelling of Kelvin-Helmholtz
   isntability using smoothed particle hydrodynamics", IJNME, 2011,
   87, pp 988--1006 [SY11]

 - Joseph P. Morris "Simulating surface tension with smoothed particle
   hydrodynamics", JCP, 2000, 33, pp 333--353 [JM00]

 - Adami et al. "A new surface-tension formulation for multi-phase SPH
   using a reproducing divergence approximation", JCP 2010, 229, pp
   5011--5021 [A10]

 - X.Y.Hu, N.A. Adams. "A multi-phase SPH method for macroscopic and
   mesoscopic flows", JCP 2006, 213, pp 844-861 [XA06]

"""
from pysph.sph.equation import Equation

from math import sqrt

from pysph.sph.equation import Group, Equation

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.sph.wc.transport_velocity import SummationDensity, \
    MomentumEquationPressureGradient, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity, \
    SolidWallNoSlipBC

from pysph.sph.wc.linalg import gj_solve, augmented_matrix


from pysph.sph.wc.basic import TaitEOS


class SurfaceForceAdami(Equation):
    def initialize(self, d_au, d_av, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_au, d_av, d_aw, d_idx, d_m, DWIJ, d_pi00, d_pi01, d_pi02,
             d_pi10, d_pi11, d_pi12, d_pi20, d_pi21, d_pi22, s_pi00, s_pi01,
             s_pi02, s_pi10, s_pi11, s_pi12, s_pi20, s_pi21, s_pi22, d_V, s_V,
             s_idx):
        s2 = s_V[s_idx]*s_V[s_idx]
        f00 = (d_pi00[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi00[s_idx]/s2)
        f01 = (d_pi01[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi01[s_idx]/s2)
        f02 = (d_pi02[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi02[s_idx]/s2)
        f10 = (d_pi10[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi10[s_idx]/s2)
        f11 = (d_pi11[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi11[s_idx]/s2)
        f12 = (d_pi12[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi12[s_idx]/s2)
        f20 = (d_pi20[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi20[s_idx]/s2)
        f21 = (d_pi21[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi21[s_idx]/s2)
        f22 = (d_pi22[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi22[s_idx]/s2)
        d_au[d_idx] += (DWIJ[0]*f00 + DWIJ[1]*f01 + DWIJ[2]*f02)/d_m[d_idx]
        d_av[d_idx] += (DWIJ[0]*f10 + DWIJ[1]*f11 + DWIJ[2]*f12)/d_m[d_idx]
        d_aw[d_idx] += (DWIJ[0]*f20 + DWIJ[1]*f21 + DWIJ[2]*f22)/d_m[d_idx]


class ConstructStressMatrix(Equation):

    def __init__(self, dest, sources, sigma, d=2):
        self.sigma = sigma
        self.d = d
        super(ConstructStressMatrix, self).__init__(dest, sources)

    def initialize(self, d_pi00, d_pi01, d_pi02, d_pi10, d_pi11, d_pi12,
                   d_pi20, d_pi21, d_pi22, d_cx, d_cy, d_cz, d_idx, d_N):
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]
        mod_gradc = sqrt(mod_gradc2)
        d_N[d_idx] = 0.0
        if mod_gradc > 1e-14:
            factor = self.sigma/mod_gradc
            d_pi00[d_idx] = (-d_cx[d_idx]*d_cx[d_idx] +
                             (mod_gradc2)/self.d)*factor
            d_pi01[d_idx] = -factor*d_cx[d_idx]*d_cy[d_idx]
            d_pi02[d_idx] = -factor*d_cx[d_idx]*d_cz[d_idx]
            d_pi10[d_idx] = -factor*d_cx[d_idx]*d_cy[d_idx]
            d_pi11[d_idx] = (-d_cy[d_idx]*d_cy[d_idx] +
                             (mod_gradc2)/self.d)*factor
            d_pi12[d_idx] = -factor*d_cy[d_idx]*d_cz[d_idx]
            d_pi20[d_idx] = -factor*d_cx[d_idx]*d_cz[d_idx]
            d_pi21[d_idx] = -factor*d_cy[d_idx]*d_cz[d_idx]
            d_pi22[d_idx] = (-d_cz[d_idx]*d_cz[d_idx] +
                             (mod_gradc2)/self.d)*factor
            d_N[d_idx] = 1.0
        else:
            d_pi00[d_idx] = 0.0
            d_pi01[d_idx] = 0.0
            d_pi02[d_idx] = 0.0
            d_pi10[d_idx] = 0.0
            d_pi11[d_idx] = 0.0
            d_pi12[d_idx] = 0.0
            d_pi20[d_idx] = 0.0
            d_pi21[d_idx] = 0.0
            d_pi22[d_idx] = 0.0


class ColorGradientAdami(Equation):

    def initialize(self, d_idx, d_cx, d_cy, d_cz):
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

    def loop(self, d_idx, d_cx, d_cy, d_cz, d_V, s_V, d_color, s_color, DWIJ,
             s_idx):
        c_i = d_color[d_idx]/(d_V[d_idx]*d_V[d_idx])
        c_j = s_color[s_idx]/(s_V[s_idx]*s_V[s_idx])
        factor = d_V[d_idx]*(c_i + c_j)
        d_cx[d_idx] += factor*DWIJ[0]
        d_cy[d_idx] += factor*DWIJ[1]
        d_cz[d_idx] += factor*DWIJ[2]


class MomentumEquationViscosityAdami(Equation):

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx,
             d_m, R2IJ, XIJ, EPS, VIJ, d_nu, s_nu):
        factor = 2.0*d_nu[d_idx]*s_nu[s_idx]/(d_nu[d_idx] + s_nu[s_idx])
        V_i = 1/(d_V[d_idx]*d_V[d_idx])
        V_j = 1/(s_V[s_idx]*s_V[s_idx])
        dwijdotrij = (DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2])
        dwijdotrij /= (R2IJ + EPS)
        factor = factor*(V_i + V_j)*dwijdotrij/d_m[d_idx]
        d_au[d_idx] += factor*VIJ[0]
        d_av[d_idx] += factor*VIJ[1]
        d_aw[d_idx] += factor*VIJ[2]


class MomentumEquationPressureGradientAdami(Equation):

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx,
             d_m):
        p_i = d_p[d_idx]/(d_V[d_idx]*d_V[d_idx])
        p_j = s_p[s_idx]/(s_V[s_idx]*s_V[s_idx])
        d_au[d_idx] += -(p_i+p_j)*DWIJ[0]/d_m[d_idx]
        d_av[d_idx] += -(p_i+p_j)*DWIJ[1]/d_m[d_idx]
        d_aw[d_idx] += -(p_i+p_j)*DWIJ[2]/d_m[d_idx]


class MomentumEquationViscosityMorris(Equation):

    def __init__(self, dest, sources, eta=0.01):
        self.eta = eta*eta
        super(MomentumEquationViscosityMorris, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_nu, s_nu, d_rho,
             s_rho, DWIJ, R2IJ, VIJ, HIJ, XIJ):
        r2 = R2IJ + self.eta*HIJ*HIJ
        dw = (DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2])/(r2)
        mult = s_m[s_idx]*(d_nu[d_idx] + s_nu[s_idx]) / \
            (d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += dw*mult*VIJ[0]
        d_av[d_idx] += dw*mult*VIJ[1]
        d_aw[d_idx] += dw*mult*VIJ[2]


class MomentumEquationPressureGradientMorris(Equation):

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_p, s_p, DWIJ, d_rho,
             s_rho):
        factor = -s_m[s_idx]*(d_p[d_idx] + s_p[s_idx]) / \
            (d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += factor*DWIJ[0]
        d_av[d_idx] += factor*DWIJ[1]
        d_aw[d_idx] += factor*DWIJ[2]


class InterfaceCurvatureFromDensity(Equation):

    def __init__(self, dest, sources, with_morris_correction=True):
        self.with_morris_correction = with_morris_correction

        super(InterfaceCurvatureFromDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_nx, d_ny, d_nz, s_nx, s_ny, s_nz,
             d_V, s_V, d_N, s_N, d_wij_sum, s_rho, s_m, WIJ, DWIJ):

        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        tmp = 1.0
        if self.with_morris_correction:
            tmp = min(d_N[d_idx], s_N[s_idx])

        d_wij_sum[d_idx] += tmp * s_m[s_idx]/s_rho[s_idx] * WIJ

        d_kappa[d_idx] += tmp*nijdotdwij*s_m[s_idx]/s_rho[s_idx]

    def post_loop(self, d_idx, d_wij_sum, d_nx, d_kappa):

        if self.with_morris_correction:
            if d_wij_sum[d_idx] > 1e-12:
                d_kappa[d_idx] /= d_wij_sum[d_idx]


class SolidWallPressureBCnoDensity(Equation):

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        d_p[d_idx] += s_p[s_idx]*WIJ

        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class SummationDensitySourceMass(Equation):

    def initialize(self, d_idx, d_V, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ, s_m, s_idx):
        d_rho[d_idx] += s_m[s_idx]*WIJ

    def post_loop(self, d_idx, d_V, d_rho, d_m):
        d_V[d_idx] = d_rho[d_idx]/d_m[d_idx]


class SmoothedColor(Equation):
    r"""Smoothed color function. Eq. (17) in [JM00]

    .. math::

        c_a = \sum_b \frac{m_b}{\rho_b} c_b^i \nabla_a W_{ab}\,,

    where, :math:`c_b^i` is the color index associated with a
    particle.

    """

    def initialize(self, d_idx, d_scolor):
        d_scolor[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             s_color, d_scolor, WIJ):

        # Smoothed color Eq. (17) in [JM00]
        d_scolor[d_idx] += s_m[s_idx]/s_rho[s_idx] * s_color[s_idx] * WIJ


class ColorGradientUsingNumberDensity(Equation):
    r"""Gradient of the color function using Eq. (13) of [SY11]:

    .. math::

        \nabla C_a = \sum_b \frac{2 C_b - C_a}{\psi_a + \psi_a}
        \nabla_{a} W_{ab}


    Using the gradient of the color function, the normal and
    discretized dirac delta is calculated in the post
    loop.

    Singularities are avoided as per the recommendation by [JM00] (see eqs
    20 & 21) using the parameter :math:`\epsilon`

    """

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(ColorGradientUsingNumberDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz,
                   d_ddelta, d_N):

        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # interface normals
        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # discretized dirac delta
        d_ddelta[d_idx] = 0.0

        # reliability indicator for normals
        d_N[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_scolor, s_scolor, d_cx, d_cy, d_cz,
             d_V, s_V, DWIJ):

        # average particle volume
        psiab1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # difference in color divided by psiab. Eq. (13) in [SY11]
        Cba = (s_scolor[s_idx] - d_scolor[d_idx]) * psiab1

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero) Eq. (19,
        # 20) in [JM00]
        if mod_gradc2 > self.epsilon2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1./mod_gradc


class MorrisColorGradient(Equation):
    r"""Gradient of the color function using Eq. (17) of [JM00]:

    .. math::

        \nabla c_a = \sum_b \frac{m_b}{\rho_b}(c_b - c_a) \nabla_{a}
        W_{ab}\,,

    where a smoothed representation of the color is used in the
    equation. Using the gradient of the color function, the normal and
    discretized dirac delta is calculated in the post loop.

    Singularities are avoided as per the recommendation by [JM00] (see eqs
    20 & 21) using the parameter :math:`\epsilon`

    """

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(MorrisColorGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz,
                   d_ddelta, d_N):

        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # interface normals
        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # reliability indicator for normals and dirac delta
        d_N[d_idx] = 0.0
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_scolor, s_scolor, d_cx, d_cy, d_cz,
             s_m, s_rho, DWIJ):

        # Eq. (17) in [JM00]
        Cba = (s_scolor[s_idx] - d_scolor[d_idx]) * s_m[s_idx]/s_rho[s_idx]

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero) Eq. (19,
        # 20) in [JM00]
        if mod_gradc2 > self.epsilon2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1./mod_gradc


class SY11ColorGradient(Equation):
    r"""Gradient of the color function using Eq. (13) of [SY11]:

    .. math::

        \nabla C_a = \sum_b \frac{2 C_b - C_a}{\psi_a + \psi_a}
        \nabla_{a} W_{ab}


    Using the gradient of the color function, the normal and
    discretized dirac delta is calculated in the post
    loop.

    """

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(SY11ColorGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz,
                   d_ddelta, d_N):

        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # interface normals
        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # discretized dirac delta
        d_ddelta[d_idx] = 0.0

        # reliability indicator for normals
        d_N[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_color, s_color, d_cx, d_cy, d_cz,
             d_V, s_V, DWIJ):

        # average particle volume
        psiab1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # difference in color divided by psiab. Eq. (13) in [SY11]
        Cba = (s_color[s_idx] - d_color[d_idx]) * psiab1

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        if mod_gradc2 > self.epsilon2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1./mod_gradc


class SY11DiracDelta(Equation):
    r"""Discretized dirac-delta for the SY11 formulation Eq. (14) in [SY11]

    This is essentially the same as computing the color gradient, the
    only difference being that this might be called with a reduced
    smoothing length.

    Note that the normals should be computed using the
    SY11ColorGradient equation. This function will effectively
    overwrite the color gradient.

    """

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(SY11DiracDelta, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_ddelta):
        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # discretized dirac delta
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_color, s_color, d_cx, d_cy, d_cz,
             d_V, s_V, DWIJ):

        # average particle volume
        psiab1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # difference in color divided by psiab. Eq. (13) in [SY11]
        Cba = (s_color[s_idx] - d_color[d_idx]) * psiab1

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        if mod_gradc2 > self.epsilon2:
            mod_gradc = sqrt(mod_gradc2)

            # discretized Dirac Delta function
            d_ddelta[d_idx] = mod_gradc


class InterfaceCurvatureFromNumberDensity(Equation):
    r"""Interface curvature using number density. Eq. (15) in [SY11]:

    .. math::

        \kappa_a = \sum_b \frac{2.0}{\psi_a + \psi_b}
        \left(\boldsymbol{n_a} - \boldsymbol{n_b}\right) \cdot
        \nabla_a W_{ab}

    """

    def __init__(self, dest, sources, with_morris_correction=True):
        self.with_morris_correction = with_morris_correction

        super(InterfaceCurvatureFromNumberDensity, self).__init__(dest,
                                                                  sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_nx, d_ny, d_nz, s_nx, s_ny, s_nz,
             d_V, s_V, d_N, s_N, d_wij_sum, s_rho, s_m, WIJ, DWIJ):

        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        # averaged particle number density
        psiij1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # local number density with reliable normals Eq. (24) in [JM00]
        tmp = 1.0
        if self.with_morris_correction:
            tmp = min(d_N[d_idx], s_N[s_idx])

        d_wij_sum[d_idx] += tmp * s_m[s_idx]/s_rho[s_idx] * WIJ

        # Eq. (15) in [SY11] with correction Eq. (22) in [JM00]
        d_kappa[d_idx] += tmp * psiij1 * nijdotdwij

    def post_loop(self, d_idx, d_wij_sum, d_nx, d_kappa):
        # correct the curvature estimate. Eq. (23) in [JM00]
        if self.with_morris_correction:
            if d_wij_sum[d_idx] > 1e-12:
                d_kappa[d_idx] /= d_wij_sum[d_idx]


class ShadlooYildizSurfaceTensionForce(Equation):
    r"""Acceleration due to surface tension force Eq. (7,9) in [SY11]:

    .. math:

        \frac{d\boldsymbol{v}_a} = \frac{1}{m_a} \sigma \kappa_a
        \boldsymbol{n}_a \delta_a^s\,,

    where, :math:`\delta^s` is the discretized dirac delta function,
    :math:`\boldsymbol{n}` is the interface normal, :math:`\kappa` is
    the discretized interface curvature and :math:`\sigma` is the
    surface tension force constant.

    """

    def __init__(self, dest, sources, sigma=0.1):
        self.sigma = sigma

        # base class initialization
        super(ShadlooYildizSurfaceTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, d_kappa,
             d_nx, d_ny, d_nz, d_m, d_rho, d_ddelta):

        mi = 1./d_m[d_idx]
        rhoi = 1./d_rho[d_idx]

        # acceleration per uint mass term Eq. (7) in [SY11]
        tmp = self.sigma * d_kappa[d_idx] * d_ddelta[d_idx] * rhoi

        d_au[d_idx] += tmp * d_nx[d_idx]
        d_av[d_idx] += tmp * d_ny[d_idx]
        d_aw[d_idx] += tmp * d_nz[d_idx]


class CSFSurfaceTensionForce(Equation):
    r"""Acceleration due to surface tension force Eq. (25) in [JM00]:

    .. math:

        \frac{d\boldsymbol{v}_a}{dt} = \frac{1}{rho_a} \sigma_b \kappa_a
        \boldsymbol{n}_a

    Note that as per Eq. (17) in [JM00], the un-normalized normal is
    basically the gradient of the color function. The acceleration
    term therefore depends on the gradient of the color field.

    """

    def __init__(self, dest, sources, sigma=0.1):
        self.sigma = sigma

        # base class initialization
        super(CSFSurfaceTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, d_kappa,
             d_cx, d_cy, d_cz, d_rho):

        rhoi = 1./d_rho[d_idx]

        # acceleration per uint mass term Eq. (25) in [JM00]
        tmp = self.sigma * d_kappa[d_idx] * rhoi

        d_au[d_idx] += tmp * d_cx[d_idx]
        d_av[d_idx] += tmp * d_cy[d_idx]
        d_aw[d_idx] += tmp * d_cz[d_idx]


class AdamiReproducingDivergence(Equation):
    r"""Reproducing divergence approximation Eq. (20) in [A10] to
    compute the curvature

    .. math::

        \nabla \cdot \boldsymbol{\phi}_a = d\frac{\sum_b
        \boldsymbol{\phi}_{ab}\cdot \nabla_a
        W_{ab}V_b}{\sum_b\boldsymbol{x}_{ab}\cdot \nabla_a W_{ab} V_b}

    """

    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(AdamiReproducingDivergence, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_wij_sum,
             d_nx, d_ny, d_nz, s_nx, s_ny, s_nz, d_V, s_V,
             DWIJ, XIJ, RIJ, EPS):
        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]

        # dot product in the numerator of Eq. (20)
        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        # dot product in the denominator of Eq. (20)
        xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # accumulate the contributions
        d_kappa[d_idx] += nijdotdwij * Vj
        d_wij_sum[d_idx] += xijdotdwij * Vj

    def post_loop(self, d_idx, d_kappa, d_wij_sum):
        # normalize the curvature estimate
        if d_wij_sum[d_idx] > 1e-12:
            d_kappa[d_idx] /= d_wij_sum[d_idx]
        d_kappa[d_idx] *= -self.dim


class CSFSurfaceTensionForceAdami(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_kappa, d_cx, d_cy, d_cz,
                  d_m, d_alpha, d_rho):
        d_au[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cx[d_idx]/d_rho[d_idx]
        d_av[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cy[d_idx]/d_rho[d_idx]
        d_aw[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cz[d_idx]/d_rho[d_idx]


class ShadlooViscosity(Equation):
    def __init__(self, dest, sources, alpha):
        self.alpha = alpha
        super(ShadlooViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, d_h, s_idx, s_h, d_cs, s_cs, d_rho,
             s_rho, VIJ, XIJ, d_V, s_V, R2IJ, EPS, DWIJ):
        mu1 = 0.125*self.alpha*d_h[d_idx]*d_cs[d_idx]*d_rho[d_idx]
        mu2 = 0.125*self.alpha*s_h[s_idx]*s_cs[s_idx]*s_rho[s_idx]
        mu12 = 2.0*mu1*mu2/(mu1 + mu2)
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        denominator = d_V[d_idx]*s_V[s_idx]*(R2IJ + EPS)
        piij = 8.0*mu12*vijdotxij/denominator
        d_au[d_idx] += -piij*DWIJ[0]
        d_av[d_idx] += -piij*DWIJ[1]
        d_aw[d_idx] += -piij*DWIJ[2]


class AdamiColorGradient(Equation):

    r"""Gradient of color Eq. (14) in [A10]

    .. math::

        \nabla c_a = \frac{1}{V_a}\sum_b \left[V_a^2 + V_b^2
        \right]\tilde{c}_{ab}\nabla_a W_{ab}\,,

    where, the average :math:`\tilde{c}_{ab}` is defined as

    .. math::

        \tilde{c}_{ab} = \frac{\rho_b}{\rho_a + \rho_b}c_a +
        \frac{\rho_a}{\rho_a + \rho_b}c_b

    """

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz, d_ddelta,
                   d_N):
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # reliability indicator for normals
        d_N[d_idx] = 0.0

        # Discretized dirac-delta
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_V, s_V, d_rho, s_rho,
             d_cx, d_cy, d_cz, d_color, s_color, DWIJ):

        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # averaged particle color
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        rhoij1 = 1./(rhoi + rhoj)

        # Eq. (15) in [A10]
        cij = rhoj*rhoij1*d_color[d_idx] + rhoi*rhoij1*s_color[s_idx]

        # comute the gradient
        tmp = cij * (Vi2 + Vj2)/Vi

        d_cx[d_idx] += tmp * DWIJ[0]
        d_cy[d_idx] += tmp * DWIJ[1]
        d_cz[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz, d_h,
                  d_nx, d_ny, d_nz, d_ddelta, d_N):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        h2 = d_h[d_idx]*d_h[d_idx]
        if mod_gradc2 > 1e-4/h2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            one_mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * one_mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * one_mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * one_mod_gradc

            # discretized dirac delta
            d_ddelta[d_idx] = 1./one_mod_gradc


def get_surface_tension_equations(fluids, solids, scheme, rho0, p0, c0, b,
                                  factor1, factor2, nu, sigma, d, epsilon,
                                  gamma, real=False):
    """
    This function returns the required equations for the multiphase
    formulation taking inputs of the fluid particles array, solid particles
    array, the scheme to be used and other physical parameters
    Parameters
    ------------------

    fluids: list
        List of names of fluid particle arrays
    solids: list
        List of names of solid particle arrays
    scheme: string
        The scheme with which the equations are to be setup.
        Supported Schemes:
            1. TVF scheme with Morris' surface tension.
            String to be used: "tvf"
            2. Adami's surface tension implementation which doesn't involve
            calculation of curvature. String to be used: "adami_stress"
            3. Adami's surface tension implementation which involves
            calculation of curvature. String to be used: "adami"
            4. Shadloo Yildiz surface tension formulation.
            String to be used: "shadloo"
            5. Morris' surface tension formulation. This is the default scheme
            which will be used if none of the above strings are input as
            scheme.
    rho0 : float
        The reference density of the medium (Currently multiple reference
        densities for different particles is not supported)
    p0 : float
        The background pressure of the medium(Currently multiple background
        pressures for different particles is not supported)
    c0 : float
        The speed of sound of the medium(Currently multiple speeds of sounds
        for different particles is not supported)
    b : float
        The b parameter of the generalized Tait Equation of State. Refer to
        the Tait Equation's documentation for reference
    factor1 : float
        The factor for scaling of smoothing length for calculation of
        interface curvature number for shadloo's scheme
    factor2 : float
        The factor for scaling back of smoothing length for calculation of
        forces after calculating the interface curvature number in shadloo's
        scheme
    nu : float
        The kinematic viscosity of the medium
    sigma : float
        The surface tension of the system
    d : int
        The number of dimensions of the problem in the cartesian space
    epsilon: float
        Put this option false if the equations are supposed to be evaluated
        for the ghost particles, else keep it True
    """
    if scheme == 'tvf':
        result = []
        equations = []
        for i in fluids+solids:
            equations.append(SummationDensity(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(StateEquation(dest=i, sources=None, rho0=rho0,
                                           p0=p0))
            equations.append(SmoothedColor(dest=i, sources=fluids+solids))
        for i in solids:
            equations.append(SolidWallPressureBCnoDensity(dest=i,
                                                          sources=fluids))
            equations.append(SmoothedColor(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MorrisColorGradient(dest=i, sources=fluids+solids,
                                                 epsilon=epsilon))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(InterfaceCurvatureFromNumberDensity(
                dest=i, sources=fluids+solids, with_morris_correction=True))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MomentumEquationPressureGradient(dest=i,
                             sources=fluids+solids, pb=p0))
            equations.append(MomentumEquationViscosity(dest=i, sources=fluids,
                             nu=nu))
            equations.append(CSFSurfaceTensionForce(dest=i, sources=None,
                             sigma=sigma))
            equations.append(MomentumEquationArtificialStress(dest=i,
                                                              sources=fluids))
            if len(solids) != 0:
                equations.append(SolidWallNoSlipBC(dest=i, sources=solids,
                                 nu=nu))
        result.append(Group(equations))
    elif scheme == 'adami_stress':
        result = []
        equations = []
        for i in fluids+solids:
            equations.append(SummationDensity(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(TaitEOS(dest=i, sources=None, rho0=rho0, c0=c0,
                             gamma=gamma, p0=p0))
        for i in solids:
            equations.append(SolidWallPressureBCnoDensity(dest=i,
                             sources=fluids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(ColorGradientAdami(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(ConstructStressMatrix(dest=i, sources=None,
                                                   sigma=sigma, d=d))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MomentumEquationPressureGradientAdami(dest=i,
                             sources=fluids+solids))
            equations.append(MomentumEquationViscosityAdami(dest=i,
                                                            sources=fluids))
            equations.append(SurfaceForceAdami(dest=i, sources=fluids+solids))
            if len(solids) != 0:
                equations.append(SolidWallNoSlipBC(dest=i, sources=solids,
                                                   nu=nu))
        result.append(Group(equations))
    elif scheme == 'adami':
        result = []
        equations = []
        for i in fluids+solids:
            equations.append(SummationDensity(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(StateEquation(dest=i, sources=None, rho0=rho0,
                                           p0=p0, b=b))
        for i in solids:
            equations.append(SolidWallPressureBCnoDensity(dest=i,
                                                          sources=fluids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(AdamiColorGradient(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(AdamiReproducingDivergence(dest=i,
                                                        sources=fluids+solids,
                                                        dim=d))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MomentumEquationPressureGradient(
                dest=i, sources=fluids+solids, pb=0.0))
            equations.append(MomentumEquationViscosityAdami(dest=i,
                                                            sources=fluids))
            equations.append(CSFSurfaceTensionForceAdami(dest=i, sources=None))
            if len(solids) != 0:
                equations.append(SolidWallNoSlipBC(dest=i, sources=solids,
                                 nu=nu))
        result.append(Group(equations))
    elif scheme == 'shadloo':
        result = []
        equations = []
        for i in fluids+solids:
            equations.append(SummationDensity(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(StateEquation(dest=i, sources=None, rho0=rho0,
                                           p0=p0, b=b))
            equations.append(SY11ColorGradient(dest=i, sources=fluids+solids))
        for i in solids:
            equations.append(SolidWallPressureBCnoDensity(dest=i,
                                                          sources=fluids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(ScaleSmoothingLength(dest=i, sources=None,
                                                  factor=factor1))
        result.append(Group(equations, real=real, update_nnps=True))
        equations = []
        for i in fluids:
            equations.append(SY11DiracDelta(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(InterfaceCurvatureFromNumberDensity(
                dest=i, sources=fluids+solids, with_morris_correction=True))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(ScaleSmoothingLength(dest=i, sources=None,
                                                  factor=factor2))
        result.append(Group(equations, real=real, update_nnps=True))
        equations = []
        for i in fluids:
            equations.append(MomentumEquationPressureGradient(
                dest=i, sources=fluids+solids, pb=0.0))
            equations.append(MomentumEquationViscosity(dest=i, sources=fluids,
                                                       nu=nu))
            equations.append(ShadlooYildizSurfaceTensionForce(dest=i,
                                                              sources=None,
                                                              sigma=sigma))
            if len(solids) != 0:
                equations.append(SolidWallNoSlipBC(dest=i, sources=solids,
                                                   nu=nu))
        result.append(Group(equations))
    else:
        result = []
        equations = []
        for i in fluids+solids:
            equations.append(SummationDensitySourceMass(dest=i,
                                                        sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(TaitEOS(dest=i, sources=None, rho0=rho0, c0=c0,
                                     gamma=gamma, p0=0.0))
            equations.append(SmoothedColor(dest=i, sources=fluids+solids))
        for i in solids:
            equations.append(SolidWallPressureBCnoDensity(dest=i,
                                                          sources=fluids))
            equations.append(SmoothedColor(dest=i, sources=fluids+solids))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MorrisColorGradient(dest=i, sources=fluids+solids,
                                                 epsilon=epsilon))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(InterfaceCurvatureFromDensity(
                dest=i, sources=fluids+solids, with_morris_correction=True))
        result.append(Group(equations, real=real))
        equations = []
        for i in fluids:
            equations.append(MomentumEquationPressureGradientMorris(dest=i,
                             sources=fluids+solids))
            equations.append(MomentumEquationViscosityMorris(dest=i,
                                                             sources=fluids))
            equations.append(CSFSurfaceTensionForce(dest=i, sources=None,
                                                    sigma=sigma))
            if len(solids) != 0:
                equations.append(SolidWallNoSlipBC(dest=i, sources=solids,
                                                   nu=nu))
        result.append(Group(equations))
    return result
