"""
Basic Equations for Solid Mechanics
###################################

References
----------
.. [Gray2001] J. P. Gray et al., "SPH elastic dynamics", Computer Methods
    in Applied Mechanics and Engineering, 190 (2001), pp 6641 - 6662.
"""

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from textwrap import dedent

from pysph.base.utils import get_particle_array
import numpy as np


def get_bulk_mod(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0 * G * (1 + nu) / (3 * (1 - 2 * nu))


def get_speed_of_sound(E, nu, rho0):
    return np.sqrt(E / (3 * (1. - 2 * nu) * rho0))


def get_shear_modulus(E, nu):
    return E / (2. * (1. + nu))


def get_particle_array_elastic_dynamics(constants=None, **props):
    """Return a particle array for the Standard SPH formulation of
    solids.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """

    solids_props = [
        'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12', 'v20', 'v21',
        'v22', 'r00', 'r01', 'r02', 'r11', 'r12', 'r22', 's00', 's01', 's02',
        's11', 's12', 's22', 'as00', 'as01', 'as02', 'as11', 'as12', 'as22',
        's000', 's010', 's020', 's110', 's120', 's220', 'arho', 'au', 'av',
        'aw', 'ax', 'ay', 'az', 'ae', 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0',
        'z0', 'e0'
    ]

    # set wdeltap to -1. Which defaults to no self correction
    consts = {
        'wdeltap': -1.,
        'n': 4,
        'G': 0.0,
        'E': 0.0,
        'nu': 0.0,
        'rho_ref': 1000.0,
        'c0_ref': 0.0
    }
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=solids_props,
                            **props)

    # set the shear modulus G
    pa.G[0] = get_shear_modulus(pa.E[0], pa.nu[0])

    # set the speed of sound
    pa.cs = np.ones_like(pa.x) * get_speed_of_sound(pa.E[0], pa.nu[0],
                                                    pa.rho_ref[0])
    pa.c0_ref[0] = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class IsothermalEOS(Equation):
    r""" Compute the pressure using the Isothermal equation of state:

    :math:`p = p_0 + c_0^2(\rho_0 - \rho)`

    """

    def loop(self, d_idx, d_rho, d_p, d_c0_ref, d_rho_ref):
        d_p[d_idx] = d_c0_ref[0] * d_c0_ref[0] * (d_rho[d_idx] - d_rho_ref[0])


class MonaghanArtificialStress(Equation):
    r"""**Artificial stress to remove tensile instability**

    The dispersion relations in [Gray2001] are used to determine the
    different components of :math:`R`.

    Angle of rotation for particle :math:`a`

    .. math::

        \tan{2 \theta_a} = \frac{2\sigma_a^{xy}}{\sigma_a^{xx} - \sigma_a^{yy}}

    In rotated frame, the new components of the stress tensor are

    .. math::

        \bar{\sigma}_a^{xx} = \cos^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \sin^2{\theta_a}\sigma_a^{yy}\\

        \bar{\sigma}_a^{yy} = \sin^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \cos^2{\theta_a}\sigma_a^{yy}

    Components of :math:`R` in rotated frame:

    .. math::

        \bar{R}_{a}^{xx}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{xx}}
        {\rho^{2}} & \bar{\sigma}_{a}^{xx}>0\\0 & \bar{\sigma}_{a}^{xx}\leq0
        \end{cases}\\

        \bar{R}_{a}^{yy}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{yy}}
        {\rho^{2}} & \bar{\sigma}_{a}^{yy}>0\\0 & \bar{\sigma}_{a}^{yy}\leq0
        \end{cases}

    Components of :math:`R` in original frame:

    .. math::

        R_a^{xx} = \cos^2{\theta_a} \bar{R}_a^{xx} +
        \sin^2{\theta_a} \bar{R}_a^{yy}\\

        R_a^{yy} = \sin^2{\theta_a} \bar{R}_a^{xx} +
        \cos^2{\theta_a} \bar{R}_a^{yy}\\

        R_a^{xy} = \sin{\theta_a} \cos{\theta_a}\left(\bar{R}_a^{xx} -
        \bar{R}_a^{yy}\right)
    """

    def __init__(self, dest, sources, eps=0.3):
        r"""
        Parameters
        ----------
        eps : float
            constant
        """
        self.eps = eps
        super(MonaghanArtificialStress, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent("""
        cimport cython
        from pysph.base.linalg3 cimport eigen_decomposition
        from pysph.base.linalg3 cimport transform_diag_inv
        """)
        return code

    def loop(self, d_idx, d_rho, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             d_r00, d_r01, d_r02, d_r11, d_r12, d_r22):
        r"""Compute the stress terms

        Parameters
        ----------
        d_sxx : DoubleArray
            Stress Tensor Deviatoric components (Symmetric)

        d_rxx : DoubleArray
            Artificial stress components (Symmetric)
        """
        # 1/rho_a^2
        rhoi = d_rho[d_idx]
        rhoi21 = 1. / (rhoi * rhoi)

        ## Matrix and vector declarations ##

        # Matrix of Eigenvectors (columns)
        R = declare('matrix((3,3))')

        # Artificial stress in the original coordinates
        Rab = declare('matrix((3,3))')

        # Stress tensor with pressure.
        S = declare('matrix((3,3))')
        # Eigenvalues
        V = declare('matrix((3,))')

        # Artificial stress in principle direction
        rd = declare('matrix((3,))')

        # get the diagonal terms for the stress tensor adding pressure
        S[0][0] = d_s00[d_idx] - d_p[d_idx]
        S[1][1] = d_s11[d_idx] - d_p[d_idx]
        S[2][2] = d_s22[d_idx] - d_p[d_idx]

        S[1][2] = d_s12[d_idx]
        S[2][1] = d_s12[d_idx]
        S[0][2] = d_s02[d_idx]
        S[2][0] = d_s02[d_idx]
        S[0][1] = d_s01[d_idx]
        S[1][0] = d_s01[d_idx]

        # compute the principle stresses
        eigen_decomposition(S, R, cython.address(V[0]))

        # artificial stress corrections
        if V[0] > 0:
            rd[0] = -self.eps * V[0] * rhoi21
        else:
            rd[0] = 0

        if V[1] > 0:
            rd[1] = -self.eps * V[1] * rhoi21
        else:
            rd[1] = 0

        if V[2] > 0:
            rd[2] = -self.eps * V[2] * rhoi21
        else:
            rd[2] = 0

        # transform artificial stresses in original frame
        transform_diag_inv(cython.address(rd[0]), R, Rab)

        # store the values
        d_r00[d_idx] = Rab[0][0]
        d_r11[d_idx] = Rab[1][1]
        d_r22[d_idx] = Rab[2][2]
        d_r12[d_idx] = Rab[1][2]
        d_r02[d_idx] = Rab[0][2]
        d_r01[d_idx] = Rab[0][1]


class MomentumEquationWithStress(Equation):
    r"""**Momentum Equation with Artificial Stress**

    .. math::

        \frac{D\vec{v_a}^i}{Dt} = \sum_b m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2}
        +\frac{\sigma_b^{ij}}{\rho_b^2} + R_{ab}^{ij}f^n \right)\nabla_a W_{ab}

    where

    .. math::

        f_{ab} = \frac{W(r_{ab})}{W(\Delta p)}\\

        R_{ab}^{ij} = R_{a}^{ij} + R_{b}^{ij}
    """

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p, d_s00, d_s01,
             d_s02, d_s11, d_s12, d_s22, s_s00, s_s01, s_s02, s_s11, s_s12,
             s_s22, d_r00, d_r01, d_r02, d_r11, d_r12, d_r22, s_r00, s_r01,
             s_r02, s_r11, s_r12, s_r22, d_au, d_av, d_aw, d_wdeltap, d_n, WIJ,
             DWIJ):

        pa = d_p[d_idx]
        pb = s_p[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        rhoa21 = 1. / (rhoa * rhoa)
        rhob21 = 1. / (rhob * rhob)

        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        s00b = s_s00[s_idx]
        s01b = s_s01[s_idx]
        s02b = s_s02[s_idx]

        s10b = s_s01[s_idx]
        s11b = s_s11[s_idx]
        s12b = s_s12[s_idx]

        s20b = s_s02[s_idx]
        s21b = s_s12[s_idx]
        s22b = s_s22[s_idx]

        r00a = d_r00[d_idx]
        r01a = d_r01[d_idx]
        r02a = d_r02[d_idx]

        r10a = d_r01[d_idx]
        r11a = d_r11[d_idx]
        r12a = d_r12[d_idx]

        r20a = d_r02[d_idx]
        r21a = d_r12[d_idx]
        r22a = d_r22[d_idx]

        r00b = s_r00[s_idx]
        r01b = s_r01[s_idx]
        r02b = s_r02[s_idx]

        r10b = s_r01[s_idx]
        r11b = s_r11[s_idx]
        r12b = s_r12[s_idx]

        r20b = s_r02[s_idx]
        r21b = s_r12[s_idx]
        r22b = s_r22[s_idx]

        # Add pressure to the deviatoric components
        s00a = s00a - pa
        s00b = s00b - pb

        s11a = s11a - pa
        s11b = s11b - pb

        s22a = s22a - pa
        s22b = s22b - pb

        # compute the kernel correction term
        # if wdeltap is less than zero then no correction
        # needed
        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

            art_stress00 = fab * (r00a + r00b)
            art_stress01 = fab * (r01a + r01b)
            art_stress02 = fab * (r02a + r02b)

            art_stress10 = art_stress01
            art_stress11 = fab * (r11a + r11b)
            art_stress12 = fab * (r12a + r12b)

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = fab * (r22a + r22b)
        else:
            art_stress00 = 0.0
            art_stress01 = 0.0
            art_stress02 = 0.0

            art_stress10 = art_stress01
            art_stress11 = 0.0
            art_stress12 = 0.0

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = 0.0

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += (
            mb * (s00a * rhoa21 + s00b * rhob21 + art_stress00) * DWIJ[0] +
            mb * (s01a * rhoa21 + s01b * rhob21 + art_stress01) * DWIJ[1] +
            mb * (s02a * rhoa21 + s02b * rhob21 + art_stress02) * DWIJ[2])

        d_av[d_idx] += (
            mb * (s10a * rhoa21 + s10b * rhob21 + art_stress10) * DWIJ[0] +
            mb * (s11a * rhoa21 + s11b * rhob21 + art_stress11) * DWIJ[1] +
            mb * (s12a * rhoa21 + s12b * rhob21 + art_stress12) * DWIJ[2])

        d_aw[d_idx] += (
            mb * (s20a * rhoa21 + s20b * rhob21 + art_stress20) * DWIJ[0] +
            mb * (s21a * rhoa21 + s21b * rhob21 + art_stress21) * DWIJ[1] +
            mb * (s22a * rhoa21 + s22b * rhob21 + art_stress22) * DWIJ[2])


class HookesDeviatoricStressRate(Equation):
    r""" **Rate of change of stress **

    .. math::
        \frac{dS^{ij}}{dt} = 2\mu\left(\epsilon^{ij} - \frac{1}{3}\delta^{ij}
        \epsilon^{ij}\right) + S^{ik}\Omega^{jk} + \Omega^{ik}S^{kj}

    where

    .. math::

        \epsilon^{ij} = \frac{1}{2}\left(\frac{\partial v^i}{\partial x^j} +
        \frac{\partial v^j}{\partial x^i}\right)\\

        \Omega^{ij} = \frac{1}{2}\left(\frac{\partial v^i}{\partial x^j} -
           \frac{\partial v^j}{\partial x^i} \right)

    """

    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12,
                   d_as22):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as02[d_idx] = 0.0

        d_as11[d_idx] = 0.0
        d_as12[d_idx] = 0.0

        d_as22[d_idx] = 0.0

    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_v00,
             d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_v22, d_as00,
             d_as01, d_as02, d_as11, d_as12, d_as22, d_G):

        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v02 = d_v02[d_idx]

        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]
        v12 = d_v12[d_idx]

        v20 = d_v20[d_idx]
        v21 = d_v21[d_idx]
        v22 = d_v22[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s02 = d_s02[d_idx]

        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]
        s12 = d_s12[d_idx]

        s20 = d_s02[d_idx]
        s21 = d_s12[d_idx]
        s22 = d_s22[d_idx]

        # strain rate tensor is symmetric
        eps00 = v00
        eps01 = 0.5 * (v01 + v10)
        eps02 = 0.5 * (v02 + v20)

        eps10 = eps01
        eps11 = v11
        eps12 = 0.5 * (v12 + v21)

        eps20 = eps02
        eps21 = eps12
        eps22 = v22

        # rotation tensor is asymmetric
        omega00 = 0.0
        omega01 = 0.5 * (v01 - v10)
        omega02 = 0.5 * (v02 - v20)

        omega10 = -omega01
        omega11 = 0.0
        omega12 = 0.5 * (v12 - v21)

        omega20 = -omega02
        omega21 = -omega12
        omega22 = 0.0

        tmp = 2.0 * d_G[0]
        trace = 1.0 / 3.0 * (eps00 + eps11 + eps22)

        # S_00
        d_as00[d_idx] = tmp*( eps00 - trace ) + \
                        ( s00*omega00 + s01*omega01 + s02*omega02) + \
                        ( s00*omega00 + s10*omega01 + s20*omega02)

        # S_01
        d_as01[d_idx] = tmp*(eps01) + \
                        ( s00*omega10 + s01*omega11 + s02*omega12) + \
                        ( s01*omega00 + s11*omega01 + s21*omega02)

        # S_02
        d_as02[d_idx] = tmp*eps02 + \
                        (s00*omega20 + s01*omega21 + s02*omega22) + \
                        (s02*omega00 + s12*omega01 + s22*omega02)

        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
                        (s10*omega10 + s11*omega11 + s12*omega12) + \
                        (s01*omega10 + s11*omega11 + s21*omega12)

        # S_12
        d_as12[d_idx] = tmp*eps12 + \
                        (s10*omega20 + s11*omega21 + s12*omega22) + \
                        (s02*omega10 + s12*omega11 + s22*omega12)

        # S_22
        d_as22[d_idx] = tmp*(eps22 - trace) + \
                        (s20*omega20 + s21*omega21 + s22*omega22) + \
                        (s02*omega20 + s12*omega21 + s22*omega22)


class EnergyEquationWithStress(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0, eta=0.01):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = float(eta)
        super(EnergyEquationWithStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ae):
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p, d_cs, s_cs, d_ae,
             XIJ, VIJ, DWIJ, HIJ, R2IJ, RHOIJ1):

        rhoa = d_rho[d_idx]
        ca = d_cs[d_idx]
        pa = d_p[d_idx]

        rhob = s_rho[s_idx]
        cb = s_cs[s_idx]
        pb = s_p[s_idx]
        mb = s_m[s_idx]

        rhoa2 = 1. / (rhoa * rhoa)
        rhob2 = 1. / (rhob * rhob)

        # artificial viscosity
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij) / (R2IJ + self.eta * self.eta * HIJ * HIJ)

            piij = -self.alpha * cij * muij + self.beta * muij * muij
            piij = piij * RHOIJ1

        vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]

        # thermal energy contribution
        d_ae[d_idx] += 0.5 * mb * (pa * rhoa2 + pb * rhob2 + piij)

    def post_loop(self, d_idx, d_rho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_v00, d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21,
                  d_v22, d_ae):

        # particle density
        rhoa = d_rho[d_idx]

        # deviatoric stress rate (symmetric)
        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        # strain rate tensor (symmetric)
        eps00 = d_v00[d_idx]
        eps01 = 0.5 * (d_v01[d_idx] + d_v10[d_idx])
        eps02 = 0.5 * (d_v02[d_idx] + d_v20[d_idx])

        eps10 = eps01
        eps11 = d_v11[d_idx]
        eps12 = 0.5 * (d_v12[d_idx] + d_v21[d_idx])

        eps20 = eps02
        eps21 = eps12
        eps22 = d_v22[d_idx]

        # energy accelerations
        #sdoteij = s00a*eps00 +  s01a*eps01 + s10a*eps10 + s11a*eps11
        sdoteij = (s00a * eps00 + s01a * eps01 + s02a * eps02 + s10a * eps10 +
                   s11a * eps11 + s12a * eps12 + s20a * eps20 + s21a * eps21 +
                   s22a * eps22)

        d_ae[d_idx] += 1. / rhoa * sdoteij


class ElasticSolidsScheme(Scheme):
    def __init__(self, elastic_solids, solids, dim, artificial_stress_eps=0.3,
                 xsph_eps=0.5, alpha=1.0, beta=1.0):
        self.elastic_solids = elastic_solids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.alpha = alpha
        self.beta = beta
        self.xsph_eps = xsph_eps
        self.artificial_stress_eps = artificial_stress_eps

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (
            ContinuityEquation, MonaghanArtificialViscosity, XSPHCorrection,
            VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (
            IsothermalEOS, MomentumEquationWithStress,
            HookesDeviatoricStressRate, MonaghanArtificialStress)

        equations = []
        g1 = []
        all = self.solids + self.elastic_solids
        for elastic_solid in self.elastic_solids:
            g1.append(
                # p
                IsothermalEOS(elastic_solid, sources=None))
            g1.append(
                # vi,j : requires properties v00, v01, v10, v11
                VelocityGradient2D(dest=elastic_solid, sources=all))
            g1.append(
                # rij : requires properties r00, r01, r02, r11, r12, r22,
                #                           s00, s01, s02, s11, s12, s22
                MonaghanArtificialStress(dest=elastic_solid, sources=None,
                                         eps=self.artificial_stress_eps))

        equations.append(Group(equations=g1))

        g2 = []
        for elastic_solid in self.elastic_solids:
            g2.append(ContinuityEquation(dest=elastic_solid, sources=all), )
            g2.append(
                # au, av
                MomentumEquationWithStress(dest=elastic_solid, sources=all), )
            g2.append(
                # au, av
                MonaghanArtificialViscosity(dest=elastic_solid, sources=all,
                                            alpha=self.alpha,
                                            beta=self.beta), )
            g2.append(
                # a_s00, a_s01, a_s11
                HookesDeviatoricStressRate(dest=elastic_solid, sources=None), )
            g2.append(
                # ax, ay, az
                XSPHCorrection(dest=elastic_solid, sources=[elastic_solid],
                               eps=self.xsph_eps), )
        equations.append(Group(g2))

        return equations

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EPECIntegrator
        from pysph.sph.integrator_step import SolidMechStep

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        step_cls = SolidMechStep
        for name in self.elastic_solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)
