"""
Implementation of the equations used for surface tension modelling,
for example in KHI simulations. The references are as under:

 - M. Shadloo, M. Yildiz, "Numerical modelling of Kelvin-Helmholtz
   isntability using smoothed particle hydrodynamics", IJNME, 2011,
   87, pp 988--1006 [SY11]

"""
from pysph.sph.equation import Equation

from math import sqrt

class ColorGradientUsingNumberDensity(Equation):
    """Gradient of the color function using Eq. (13) of [SY11]:

    .. math::

        \nabla C_a = \sum_b \frac{2 C_b - C_a}{\psi_a + \psi_a}
        \nabla_{a} W_{ab}

    """
    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz):

        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # interface normals
        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_color, s_color, d_cx, d_cy, d_cz,
             d_V, s_V, DWIJ):
        
        # average particle volume
        psiab1 = 2.0/( d_V[d_idx] + s_V[s_idx] )

        # difference in color divided by psiab
        Cba = (s_color[s_idx] - d_color[d_idx]) * psiab1
        
        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero)
        if mod_gradc2 > 1e-14:
            mod_gradc = 1./sqrt( mod_gradc2 )

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

class DiscretizedDiracDelta(Equation):
    r"""Gradient of the color function using Eq. (14) of [SY11]:

    .. math::

        \delta^s = |\nabla C|_{0.8h} = \sum_b \frac{2C_{ab}}{(\psi_a +
        \psi_b)}\nabla_{a} W(\boldsymbol{r}_{ab}, 0.8h)

     Notes:

     In the post loop of this equation, the discretized version of the
     dirac-delta function and the interface normals are computed.

    """
    def initialize(self, d_idx, d_ddelta, d_cx2, d_cy2, d_cz2):

        # color gradients
        d_cx2[d_idx] = 0.0
        d_cy2[d_idx] = 0.0
        d_cz2[d_idx] = 0.0
        
        # dirac delta
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_color, s_color, d_cx2, d_cy2, d_cz2,
             d_V, s_V, DWIJ):
        
        # average particle volume
        psiab1 = 2.0/( d_V[d_idx] + s_V[s_idx] )

        # difference in color divided by psiab
        Cba = (s_color[s_idx] - d_color[d_idx]) * psiab1
        
        d_cx2[d_idx] += Cba * DWIJ[0]
        d_cy2[d_idx] += Cba * DWIJ[1]
        d_cz2[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_ddelta, d_cx2, d_cy2, d_cz2):
        # absolute value of the color gradient
        mod_gradc2 = d_cx2[d_idx]*d_cx2[d_idx] + \
            d_cy2[d_idx]*d_cy2[d_idx] + \
            d_cz2[d_idx]*d_cz2[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero)
        if mod_gradc2 > 1e-14:
            d_ddelta[d_idx] = sqrt( mod_gradc2 )

class InterfaceCurvatureFromNumberDensity(Equation):
    """Interface curvature using number density. Eq. (15) in [SY11]:

    .. math::

        \kappa_a = \sum_b \frac{2.0}{\psi_a + \psi_b}
        \left(\boldsymbol{n_a} - \boldsymbol{n_b}\right) \cdot
        \nabla_a W_{ab}

    """
    def initialize(self, d_idx, d_kappa):
        d_kappa[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_nx, d_ny, d_nz, s_nx, s_ny, s_nz, 
             d_V, s_V, DWIJ):
        
        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]
        
        # averaged particle number density
        psiij1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # Eq. (15) in [SY11]
        d_kappa[d_idx] += psiij1 * nijdotdwij

class ShadlooYildizSurfaceTensionForce(Equation):
    """Acceleration due to surface tension force Eq. (7,9) in [SY11]:

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
