"""
Basic SPH Equations
###################
"""

from pysph.sph.equation import Equation

class SummationDensity(Equation):
    r"""Good old Summation density:

    :math:`\rho_a = \sum_b m_b W_{ab}`

    """
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ

class BodyForce(Equation):
    r"""Add a body force to the particles:

    :math:`\boldsymbol{f} = f_x, f_y, f_z`

    """
    def __init__(self, dest, sources, fx=0.0, fy=0.0, fz=0.0):

        r"""
        Parameters
        ----------
        fx : float
            Body force per unit mass along the x-axis
        fy : float
            Body force per unit mass along the y-axis
        fz : float
            Body force per unit mass along the z-axis
        """

        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.fx
        d_av[d_idx] += self.fy
        d_aw[d_idx] += self.fz

class VelocityGradient2D(Equation):
    r""" Compute the SPH evaluation for the velocity gradient tensor in 2D.

    The expression for the velocity gradient is:

    :math:`\frac{\partial v^i}{\partial x^j} = \sum_{b}\frac{m_b}{\rho_b}(v_b
    - v_a)\frac{\partial W_{ab}}{\partial x_a^j}`

    Notes
    -----
    The tensor properties are stored in the variables v_ij where 'i'
    refers to the velocity component and 'j' refers to the spatial
    component. Thus v_21 is :math:`\frac{\partial v}{\partial x}`

    """
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_v00, d_v01, d_v10, d_v11,
             DWIJ, VIJ):

        tmp = s_m[s_idx]/s_rho[s_idx]

        d_v00[d_idx] += tmp * -VIJ[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -VIJ[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -VIJ[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -VIJ[1] * DWIJ[1]

class IsothermalEOS(Equation):
    r""" Compute the pressure using the Isothermal equation of state:

    :math:`p = p_0 + c_0^2(\rho_0 - \rho)`

    """
    def __init__(self, dest, sources, rho0, c0, p0):

        r"""
        Parameters
        ----------
        rho0 : float
            Reference density of the fluid (:math:`\rho_0`)
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        p0 : float
            Reference pressure in the system (:math:`p0`)
        """

        self.rho0 = rho0
        self.c0 = c0
        self.c02 = c0 * c0
        self.p0 = p0
        super(IsothermalEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        d_p[d_idx] = self.p0 + self.c02 * (d_rho[d_idx] - self.rho0)

class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij

class MonaghanArtificialViscosity(Equation):
    r"""Classical Monaghan style artificial viscosity [Monaghan2005]_

    .. math::

        \frac{d\mathbf{v}_{a}}{dt}&=&-\sum_{b}m_{b}\Pi_{ab}\nabla_{a}W_{ab}

    where

    .. math::

        \Pi_{ab}=\begin{cases}\frac{-\alpha_{\pi}\bar{c}_{ab}\phi_{ab}+
        \beta_{\pi}\phi_{ab}^{2}}{\bar{\rho}_{ab}}, & \mathbf{v}_{ab}\cdot
        \mathbf{r}_{ab}<0\\0, & \mathbf{v}_{ab}\cdot\mathbf{r}_{ab}\geq0
        \end{cases}

    with

    .. math::

        \phi_{ab}=\frac{h\mathbf{v}_{ab}\cdot\mathbf{r}_{ab}}
        {|\mathbf{r}_{ab}|^{2}+\epsilon^{2}}\\

        \bar{c}_{ab}&=&\frac{c_{a}+c_{b}}{2}\\

        \bar{\rho}_{ab}&=&\frac{\rho_{a}+\rho_{b}}{2}

    References
    ----------
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics",
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759.
    """
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        r"""
        Parameters
        ----------
        alpha : float
            produces a shear and bulk viscosity
        beta : float
            used to handle high Mach number shocks
        """
        self.alpha = alpha
        self.beta = beta
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ, DT_ADAPT):

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]

class XSPHCorrection(Equation):
    r"""Position stepping with XSPH correction [Monaghan1992]_

    .. math::

        \frac{d\mathbf{r}_{a}}{dt}=\mathbf{\hat{v}}_{a}=\mathbf{v}_{a}-
        \epsilon\sum_{b}m_{b}\frac{\mathbf{v}_{ab}}{\bar{\rho}_{ab}}W_{ab}

    References
    ----------
    .. [Monaghan1992] J. Monaghan, Smoothed Particle Hydrodynamics, "Annual
        Review of Astronomy and Astrophysics", 30 (1992), pp. 543-574.
    """
    def __init__(self, dest, sources, eps=0.5):
        r"""
        Parameters
        ----------
        eps : float
            :math:`\epsilon` as in the above equation

        Notes
        -----
        This equation must be used to advect the particles. XSPH can be
        turned off by setting the parameter ``eps = 0``.
        """

        self.eps = eps
        super(XSPHCorrection, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ax, d_ay, d_az):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0
        d_az[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, d_az, WIJ, RHOIJ1, VIJ):
        tmp = -self.eps * s_m[s_idx]*WIJ*RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]
        d_az[d_idx] += tmp * VIJ[2]

    def post_loop(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]
        d_az[d_idx] += d_w[d_idx]


class XSPHCorrectionForLeapFrog(Equation):
    r"""The XSPH correction [Monaghan1992]_ alone.  This is meant to be used
    with a leap-frog integrator which already considers the velocity of the
    particles.  It simply computes the correction term and adds that to ``ax,
    ay, az``.

    .. math::

        \frac{d\mathbf{r}_{a}}{dt}=\mathbf{\hat{v}}_{a}= -
        \epsilon\sum_{b}m_{b}\frac{\mathbf{v}_{ab}}{\bar{\rho}_{ab}}W_{ab}

    References
    ----------
    .. [Monaghan1992] J. Monaghan, Smoothed Particle Hydrodynamics, "Annual
        Review of Astronomy and Astrophysics", 30 (1992), pp. 543-574.
    """
    def __init__(self, dest, sources, eps=0.5):
        r"""
        Parameters
        ----------
        eps : float
            :math:`\epsilon` as in the above equation

        Notes
        -----
        This equation must be used to advect the particles. XSPH can be
        turned off by setting the parameter ``eps = 0``.
        """

        self.eps = eps
        super(XSPHCorrectionForLeapFrog, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ax, d_ay, d_az):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0
        d_az[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, d_az, WIJ, RHOIJ1, VIJ):
        tmp = -self.eps * s_m[s_idx]*WIJ*RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]
        d_az[d_idx] += tmp * VIJ[2]
