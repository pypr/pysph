"""
Implementation of the equations used for surface tension modelling,
for example in KHI simulations. The references are as under:

 - M. Shadloo, M. Yildiz, "Numerical modelling of Kelvin-Helmholtz
   isntability using smoothed particle hydrodynamics", IJNME, 2011,
   87, pp 988--1006 [SY11]

 - Joseph P. Morris "Simulating surface tension with smoothed particle
   hydrodynamics", JCP, 2000, 33, pp 333--353 [JM00]

"""
from pysph.sph.equation import Equation

from math import sqrt

class SmoothedColor(Equation):
    r"""Smoothed color function. Eq. (17) in [JM00]

    .. math::
    
        c_a = \sum_b \frac{m_b}{\rho_b} c_b^i \nabla_a W_{ab}\,,

    where, :math:`c_b^i` is the color index associated with a
    particle.

    """
    def __init__(self, dest, sources=None, smooth=False):
        self.smooth = smooth
        super(SmoothedColor, self).__init__(dest, sources)

    def initialize(self, d_idx, d_scolor):
        d_scolor[d_idx]  = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             s_color, d_scolor, WIJ):

        # Smoothed color Eq. (17) in [JM00]
        d_scolor[d_idx] += s_m[s_idx]/s_rho[s_idx] * s_color[s_idx] * WIJ

    def post_loop(self, d_idx, d_color, d_scolor):
        # overwrite the smoothed color if not needed
        if not self.smooth:
            d_scolor[d_idx] = d_color[d_idx]

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
    def __init__(self, dest, sources=None, epsilon=1e-6):
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
        psiab1 = 2.0/( d_V[d_idx] + s_V[s_idx] )

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
            mod_gradc = 1./sqrt( mod_gradc2 )

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1./mod_gradc

class InterfaceCurvatureFromNumberDensity(Equation):
    r"""Interface curvature using number density. Eq. (15) in [SY11]:

    .. math::

        \kappa_a = \sum_b \frac{2.0}{\psi_a + \psi_b}
        \left(\boldsymbol{n_a} - \boldsymbol{n_b}\right) \cdot
        \nabla_a W_{ab}

    """
    def __init__(self, dest, sources=None, with_morris_correction=False):
        self.with_morris_correction = with_morris_correction
        super(InterfaceCurvatureFromNumberDensity,self).__init__(dest, sources)

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
    def __init__(self, dest, sources=None, sigma=0.1):
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
