"""
Transport Velocity Formulation
##############################

References
----------
    .. [Adami2012] S. Adami et. al "A generalized wall boundary condition for
        smoothed particle hydrodynamics", Journal of Computational Physics
        (2012), pp. 7057--7075.

    .. [Adami2013] S. Adami et. al "A transport-velocity formulation for
        smoothed particle hydrodynamics", Journal of Computational Physics
        (2013), pp. 292--307.

"""

from pysph.sph.equation import Equation
from math import sin, cos, pi

# constants
M_PI = pi

class SummationDensity(Equation):
    r"""**Summation density with volume summation**

    In addition to the standard summation density, the number density
    for the particle is also computed. The number density is important
    for multi-phase flows to define a local particle volume
    independent of the material density.

    .. math::

        \rho_a = \sum_b m_b W_{ab}\\

        \mathcal{V}_a = \frac{1}{\sum_b W_{ab}}

    Notes
    -----
    For this equation, the destination particle array must define the
    variable `V` for particle volume.
    """

    def initialize(self, d_idx, d_V, d_rho):
        d_V[d_idx] = 0.0
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += d_m[d_idx]*WIJ

class VolumeSummation(Equation):
    """**Number density for volume computation**

    See `SummationDensity`
    """

    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

class VolumeFromMassDensity(Equation):
    """**Set the inverse volume using mass density**"""
    def loop(self, d_idx, d_V, d_rho, d_m):
        d_V[d_idx] = d_rho[d_idx]/d_m[d_idx]

class SetWallVelocity(Equation):
    r"""**Extrapolating the fluid velocity on to the wall**

    Eq. (22) in [Adami2012]:

    .. math::

        \tilde{\boldsymbol{v}}_a = \frac{\sum_b\boldsymbol{v}_b W_{ab}}
        {\sum_b W_{ab}}

    Notes
    -----
    The destination particle array for this equation should define the
    *filtered* velocity variables :math:`uf, vf, wf`.

    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, d_wij, WIJ):

        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx,
            d_ug, d_vg, d_wg, d_u, d_v, d_w):

        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]

class ContinuityEquation(Equation):
    r"""**Conservation of mass equation**

    Eq (6) in [Adami2012]:

    .. math::

        \frac{d\rho_a}{dt} = \rho_a \sum_b \frac{m_b}{\rho_b}
        \boldsymbol{v}_{ab} \cdot \nabla_a W_{ab}

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, s_m, d_rho, s_rho, VIJ, DWIJ):
        vijdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]

        d_arho[d_idx] += d_rho[d_idx] * s_m[s_idx]/s_rho[s_idx] * vijdotdwij

class StateEquation(Equation):
    r"""**Generalized Weakly Compressible Equation of State**

    .. math::

        p_a = p_0\left[ \left(\frac{\rho}{\rho_0}\right)^\gamma - b
        \right] + \mathcal{X}

    Notes
    -----
    This is the generalized Tait's equation of state and the suggested values
    in [Adami2013] are :math:`\mathcal{X} = 0`, :math:`\gamma=1` and
    :math:`b = 1`.

    The reference pressure :math:`p_0` is calculated from the artificial
    sound speed and reference density:

    .. math::

        p_0 = \frac{c^2\rho_0}{\gamma}
    """

    def __init__(self, dest, sources, p0, rho0, b=1.0):
        r"""
        Parameters
        ----------
        p0 : float
            reference pressure
        rho0 : float
            reference density
        b : float
            constant (default 1.0).
        """

        self.b=b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * ( d_rho[d_idx]/self.rho0 - self.b )

class MomentumEquationPressureGradient(Equation):
    r"""**Momentum equation for the Transport Velocity Formulation: Pressure**

    Eq. (8) in [Adami2013]:

    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
        V_b^2)\left[-\bar{p}_{ab}\nabla_a W_{ab} \right]

    where

    .. math::

        \bar{p}_{ab} = \frac{\rho_b p_a + \rho_a p_b}{\rho_a + \rho_b}
    """

    def __init__(self, dest, sources, pb, gx=0., gy=0., gz=0.,
                 tdamp=0.0):

        r"""
        Parameters
        ----------
        pb : float
            background pressure
        gx : float
            Body force per unit mass along the x-axis
        gy : float
            Body force per unit mass along the y-axis
        gz : float
            Body force per unit mass along the z-axis
        tdamp : float
            damping time

        Notes
        -----
        This equation should have the destination as fluid and sources as
        fluid and boundary particles.

        This function also computes the contribution to the background
        pressure and accelerations due to a body force or gravity.

        The body forces are damped according to Eq. (13) in [Adami2012] to avoid
        instantaneous accelerations. By default, damping is neglected.
        """

        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho,
             d_au, d_av, d_aw, d_p, s_p,
             d_auhat, d_avhat, d_awhat, d_V, s_V, DWIJ):

        # averaged pressure Eq. (7)
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]
        pi = d_p[d_idx]; pj = s_p[s_idx]

        pij = rhoj * pi + rhoi * pj
        pij /= (rhoj + rhoi)

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # inverse mass of destination particle
        mi1 = 1.0/d_m[d_idx]

        # accelerations 1st term in Eq. (8)
        tmp = -pij * mi1 * (Vi2 + Vj2)

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        # contribution due to the background pressure Eq. (13)
        tmp = -self.pb * mi1 * (Vi2 + Vj2)

        d_auhat[d_idx] += tmp * DWIJ[0]
        d_avhat[d_idx] += tmp * DWIJ[1]
        d_awhat[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, t):
        # damped accelerations due to body or external force
        damping_factor = 1.0
        if t < self.tdamp:
            damping_factor = 0.5 * ( sin((-0.5 + t/self.tdamp)*M_PI)+ 1.0 )

        d_au[d_idx] += self.gx * damping_factor
        d_av[d_idx] += self.gy * damping_factor
        d_aw[d_idx] += self.gz * damping_factor

class MomentumEquationViscosity(Equation):
    r"""**Momentum equation for the Transport Velocity Formulation: Viscosity**

    Eq. (8) in [Adami2013]:

    .. math::

           \frac{d \boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
           V_b^2)\left[ \bar{\eta}_{ab}\hat{r}_{ab}\cdot \nabla_a W_{ab}
           \frac{\boldsymbol{v}_{ab}}{|\boldsymbol{r}_{ab}|}\right]

    where

    .. math::

        \bar{\eta}_{ab} = \frac{2\eta_a \eta_b}{\eta_a + \eta_b}
    """

    def __init__(self, dest, sources, nu):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """

        self.nu = nu
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m, d_V, s_V,
             d_au, d_av, d_aw,
             R2IJ, EPS, DWIJ, VIJ, XIJ):

        # averaged shear viscosity Eq. (6)
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # scalar part of the kernel gradient
        Fij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # accelerations 3rd term in Eq. (8)
        tmp = 1./d_m[d_idx] * (Vi2 + Vj2) * etaij * Fij/(R2IJ + EPS)

        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]

class MomentumEquationArtificialViscosity(Equation):
    r"""**Artificial viscosity for the momentum equation**

    Eq. (11) in [Adami2012]:

    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = -\sum_b m_b \alpha h_{ab}
        c_{ab} \frac{\boldsymbol{v}_{ab}\cdot
        \boldsymbol{r}_{ab}}{\rho_{ab}\left(|r_{ab}|^2 + \epsilon
        \right)}\nabla_a W_{ab}

    where

    .. math::

        \rho_{ab} = \frac{\rho_a + \rho_b}{2}\\

        c_{ab} = \frac{c_a + c_b}{2}\\

        h_{ab} = \frac{h_a + h_b}{2}
    """
    def __init__(self, dest, sources, c0, alpha=0.1):
        r"""
        Parameters
        ----------
        alpha : float
            constant
        c0 : float
            speed of sound
        """

        self.alpha = alpha
        self.c0 = c0
        super(MomentumEquationArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_au, d_av, d_aw,
             RHOIJ1, R2IJ, EPS, DWIJ, VIJ, XIJ, HIJ):

        # v_{ab} \cdot r_{ab}
        vijdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        # scalar part of the accelerations Eq. (11)
        piij = 0.0
        if vijdotrij < 0:
            muij = (HIJ * vijdotrij)/(R2IJ + EPS)

            piij = -self.alpha*self.c0*muij
            piij = s_m[s_idx] * piij*RHOIJ1

        d_au[d_idx] += -piij * DWIJ[0]
        d_av[d_idx] += -piij * DWIJ[1]
        d_aw[d_idx] += -piij * DWIJ[2]


class MomentumEquationArtificialStress(Equation):
    r"""**Artificial stress contribution to the Momentum Equation**

    .. math::

          \frac{d\boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
          V_b^2)\left[ \frac{1}{2}(\boldsymbol{A}_a +
          \boldsymbol{A}_b) : \nabla_a W_{ab}\right]

    where the artificial stress terms are given by:

    .. math::

           \boldsymbol{A} = \rho \boldsymbol{v} (\tilde{\boldsymbol{v}}
         - \boldsymbol{v})

    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_u, d_v, d_w, d_V,
             d_uhat, d_vhat, d_what, d_au, d_av, d_aw, d_m,
             s_rho, s_u, s_v, s_w, s_V, s_uhat, s_vhat, s_what, DWIJ):
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]

        # physical and advection velocities
        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]
        wi = d_w[d_idx]; whati = d_what[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]
        wj = s_w[s_idx]; whatj = s_what[s_idx]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # artificial stress tensor
        Axxi = rhoi*ui*(uhati - ui); Axyi = rhoi*ui*(vhati - vi)
        Axzi = rhoi*ui*(whati - wi)
        Ayxi = rhoi*vi*(uhati - ui); Ayyi = rhoi*vi*(vhati - vi)
        Ayzi = rhoi*vi*(whati - wi)
        Azxi = rhoi*wi*(uhati - ui); Azyi = rhoi*wi*(vhati - vi)
        Azzi = rhoi*wi*(whati - wi)

        Axxj = rhoj*uj*(uhatj - uj); Axyj = rhoj*uj*(vhatj - vj)
        Axzj = rhoj*uj*(whatj - wj)
        Ayxj = rhoj*vj*(uhatj - uj); Ayyj = rhoj*vj*(vhatj - vj)
        Ayzj = rhoj*vj*(whatj - wj)
        Azxj = rhoj*wj*(uhatj - uj); Azyj = rhoj*wj*(vhatj - vj)
        Azzj = rhoj*wj*(whatj - wj)

        # contraction of stress tensor with kernel gradient
        Ax = 0.5*(
            (Axxi + Axxj)*DWIJ[0] +
            (Axyi + Axyj)*DWIJ[1] +
            (Axzi + Axzj)*DWIJ[2]
        )

        Ay = 0.5*(
            (Ayxi + Ayxj)*DWIJ[0] +
            (Ayyi + Ayyj)*DWIJ[1] +
            (Ayzi + Ayzj)*DWIJ[2]
        )

        Az = 0.5*(
            (Azxi + Azxj)*DWIJ[0] +
            (Azyi + Azyj)*DWIJ[1] +
            (Azzi + Azzj)*DWIJ[2]
        )

        # accelerations 2nd part of Eq. (8)
        tmp = 1./d_m[d_idx] * (Vi2 + Vj2)

        d_au[d_idx] += tmp * Ax
        d_av[d_idx] += tmp * Ay
        d_aw[d_idx] += tmp * Az

class SolidWallNoSlipBC(Equation):
    r"""**Solid wall boundary condition** [Adami2012]_

    This boundary condition is to be used with fixed ghost particles
    in SPH simulations and is formulated for the general case of
    moving boundaries.

    The velocity and pressure of the fluid particles is extrapolated
    to the ghost particles and these values are used in the equations
    of motion.

    No-penetration:

    Ghost particles participate in the continuity and state equations
    with fluid particles. This means as fluid particles approach the
    wall, the pressure of the ghost particles increases to generate a
    repulsion force that prevents particle penetration.

    No-slip:

    Extrapolation is used to set the `dummy` velocity of the ghost
    particles for viscous interaction. First, the smoothed velocity
    field of the fluid phase is extrapolated to the wall particles:

    .. math::

        \tilde{v}_a = \frac{\sum_b v_b W_{ab}}{\sum_b W_{ab}}

    In the second step, for the viscous interaction in Eqs. (10) in [Adami2012]
    and Eq. (8) in [Adami2013], the velocity of the ghost particles is
    assigned as:

    .. math::

       v_b = 2v_w -\tilde{v}_a,

    where :math:`v_w` is the prescribed wall velocity and :math:`v_b`
    is the ghost particle in the interaction.
    """

    def __init__(self, dest, sources, nu):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity

        Notes
        -----
        For this equation the destination particle array should be the
        fluid and the source should be ghost or boundary particles. The
        boundary particles must define a prescribed velocity :math:`u_0,
        v_0, w_0`
        """

        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, d_V, s_V,
             d_u, d_v, d_w,
             d_au, d_av, d_aw,
             s_ug, s_vg, s_wg,
             DWIJ, R2IJ, EPS, XIJ):

        # averaged shear viscosity Eq. (6).
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # scalar part of the kernel gradient
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # viscous contribution (third term) from Eq. (8), with VIJ
        # defined appropriately using the ghost values
        tmp = 1./d_m[d_idx] * (Vi2 + Vj2) * (etaij * Fij/(R2IJ + EPS))

        d_au[d_idx] += tmp * (d_u[d_idx] - s_ug[s_idx])
        d_av[d_idx] += tmp * (d_v[d_idx] - s_vg[s_idx])
        d_aw[d_idx] += tmp * (d_w[d_idx] - s_wg[s_idx])

class SolidWallPressureBC(Equation):
    r"""**Solid wall pressure boundary condition** [Adami2012]_

    This boundary condition is to be used with fixed ghost particles
    in SPH simulations and is formulated for the general case of
    moving boundaries.

    The velocity and pressure of the fluid particles is extrapolated
    to the ghost particles and these values are used in the equations
    of motion.

    Pressure boundary condition:

    The pressure of the ghost particle is also calculated from the
    fluid particle by interpolation using:

    .. math::

        p_g = \frac{\sum_f p_f W_{gf} + \boldsymbol{g - a_g} \cdot
        \sum_f \rho_f \boldsymbol{r}_{gf}W_{gf}}{\sum_f W_{gf}},

    where the subscripts `g` and `f` relate to the ghost and fluid
    particles respectively.

    Density of the wall particle is then set using this pressure

    .. math::

        \rho_w=\rho_0\left(\frac{p_w - \mathcal{X}}{p_0} +
        1\right)^{\frac{1}{\gamma}}
    """

    def __init__(self, dest, sources, rho0, p0, b=1.0, gx=0.0, gy=0.0, gz=0.0):
        r"""
        Parameters
        ----------
        rho0 : float
            reference density
        p0 : float
            reference pressure
        b : float
            constant (default 1.0)
        gx : float
            Body force per unit mass along the x-axis
        gy : float
            Body force per unit mass along the y-axis
        gz : float
            Body force per unit mass along the z-axis

        Notes
        -----
        For a two fluid system (boundary, fluid), this equation must be
        instantiated with boundary as the destination and fluid as the
        source.

        The boundary particle array must additionally define a property
        :math:`wij` for the denominator in Eq. (27) from [Adami2012]. This array
        sums the kernel terms from the ghost particle to the fluid
        particle.
        """

        self.rho0 = rho0
        self.p0 = p0
        self.b = b
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(SolidWallPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure Eq. (28)
        d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + self.b)
