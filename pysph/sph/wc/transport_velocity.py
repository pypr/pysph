"""Implementation of the Transport Velocity Formulation with
generalized wall boundary conditions of Adami et. al

The references are

 - Adami et. al "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 7057--7075 (REF1)

 - Adami et. al "A transport-velocity formulation for smoothed
   particle hydrodynamics", 2013, JCP, 292--307 (REF2)

"""

from pysph.sph.equation import Equation

class DensitySummation(Equation):
    """Summation density with volume summation

    In addition to the standard summation density, the number density
    for the particle is also computed. The number density is important
    for multi-phase flows to define a local particle volume
    independent of the material density.

    Notes:
    
    The volume for the particle, computed from the number density is
    given as:

    .. math::

        \mathcal{V}_a = \frac{1}{\sum_b W_{ab}}

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
    """Number density for volume computation.

    See `DensitySummation`

    """
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

class ShepardFilteredVelocity(Equation):
    """Shepard filtered smooth velocity Eq. (22) in REF2:
    
    .. math::

        \tlide{\boldsymbol{v}}_a = \frac{1}{V_a}\sum_b
        \boldsymbol{v}_b W_{ab},

    where :math:`V` is the particle volume computed through either
    `VolumeSummation` or `DensitySummation`.

    Notes:

    The destination particle array for this equation should define the
    *filtered* velocity variables :math:`uf, vf, wf`.

    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, d_V, WIJ):

        # particle volume
        Vi = 1./d_V[d_idx]
        
        # sum in Eq. (22)
        d_uf[d_idx] += Vi * s_u[s_idx] * WIJ
        d_vf[d_idx] += Vi * s_v[s_idx] * WIJ
        d_wf[d_idx] += Vi * s_w[s_idx] * WIJ

class ContinuityEquation(Equation):
    """Conservation of mass equation Eq (6) in REF1

    .. math::

        \frac{d\rho_a}{dt} = \rho_a \sum_b \frac{m_b}{\rho_b}
        \boldsymbol{v}_{ab} \cdot \nabla_a W_{ab}

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, d_m, s_V, VIJ, DWIJ):
        Vj = 1./s_V[s_idx]
        vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]
        d_arho[d_idx] += d_m[d_idx] * Vj * vijdotdwij

class StateEquation(Equation):
    """Generalized weakly compressible EOS:

    .. math::

        p_a = p_0\left[ \left(\frac{\rho}{\rho_0}\right)^\gamma - b
        \right] + \mathcal{X}

     Notes:
     
     This is the generalized Tait's equation of state and the
     suggested values in REF2 are :math:`\mathcal{X} = 0`,
     :math:`\gamma=1` and :math:`b = 1`

     The reference pressure :math:`p_0` is calculated from the
     artificial sound speed and reference density:
     
     .. math::
      
         p_0 = \frac{c^2\rho_0}{\gamma}

    """
    def __init__(self, dest, sources=None, p0=1.0, rho0=1.0, b=1.0):
        self.b=b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * ( d_rho[d_idx]/self.rho0 - self.b )

class MomentumEquationPressureGradient(Equation):
    """Momentum equation for the Transport Velocity formulation
    Eq. (8) in REF2:

    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
        V_b^2)\left[-\bar{p}_{ab}\nabla_a W_{ab} \right]

    Notes:
    
    This equation should have the destination as fluid and sources as
    fluid and boundary particles.

    This function also computes the contribution to the background
    pressure.

    """
    def __init__(self, dest, sources=None, pb=0.0, gx=0., gy=0., gz=0.):
        self.pb = 0.0
        self.gx = gx
        self.gy = gy
        self.gz = gz
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

    def post_loop(self, d_idx, d_au, d_av, d_aw):
        # acceleration due to gravity or external body force
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz

class MomentumEquationViscosity(Equation):
    """Momentum equation for the Transport Velocity formulation
    Eq. (8) in REF2:

    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
        V_b^2)\left[ \bar{\eta}_{ab}\hat{r}_{ab}\cdot \nabla_a W_{ab}
        \frac{\boldsymbol{v}_{ab}}{|\boldsymbol{r}_{ab}|}\right]

    """
    def __init__(self, dest, sources=None, nu=0.01):
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

class MomentumEquationArtificialStress(Equation):
    """Artificial stress contribution to the Momentum Equation

    .. math::

        \frac{d\boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +
        V_b^2)\left[ \frac{1}{2}(\boldsymbol{A}_a +
        \boldsymbol{A}_b) : \nabla_a W_{ab}\right]

     where the artificial stress terms are given by:

     .. math::

         \boldsymbol{A} = \rho \boldsymbol{v} (\tilde{\boldsymbol{v}}
         - \boldsymbol{v})

    """
    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_u, d_v, d_V, d_uhat, d_vhat,
             d_au, d_av, d_m, s_rho, s_u, s_v, s_V, s_uhat, s_vhat,
             DWIJ):
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]

        # physical and advection velocities
        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # artificial stress tensor
        Axxi = rhoi*ui*(uhati - ui); Axyi = rhoi*ui*(vhati - vi)
        Ayxi = rhoi*vi*(uhati - ui); Ayyi = rhoi*vi*(vhati - vi)

        Axxj = rhoj*uj*(uhatj - uj); Axyj = rhoj*uj*(vhatj - vj)
        Ayxj = rhoj*vj*(uhatj - uj); Ayyj = rhoj*vj*(vhatj - vj)

        # contraction of stress tensor with kernel gradient
        Ax = 0.5 * (Axxi + Axxj) * DWIJ[0] + 0.5 * (Axyi + Axyj) * DWIJ[1]
        Ay = 0.5 * (Ayxi + Ayxj) * DWIJ[0] + 0.5 * (Ayyi + Ayyj) * DWIJ[1]

        # accelerations 2nd part of Eq. (8)
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ax
        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ay

class SolidWallNoSlipBC(Equation):
    """Solid wall boundary condition described in REF1
    
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
    particles for viscous interaction. First, a Shepard filtered
    velocity is computed for fluid particles in the vicinity of the
    wall:

    .. math::
    
        \tilde{v}_a = \frac{\sum_b v_b W_{ab}}{\sum_b W_{ab}}

    In the second step, for the viscous interaction in Eqs. (10) in
    REF1 and Eq. (8) in REF2, the velocity of the ghost particles is
    temorarily assigned as:

    .. math::

       v_b = 2v_w -\tilde{v}_a,

    where :math:`v_w` is the prescribed wall velocity and :math:`v_b`
    is the ghost particle in the interaction.

    Notes:
    
    For this equation the destination particle array should be the
    fluid and the source should be ghost or boundary particles. The
    boundary particles must define a prescribed velocity :math:`u_0,
    v_0, w_0`
    
    """
    def __init__(self, dest, sources=None, nu=0.01):
        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, d_V, s_V,
             d_u, d_v, d_w, d_uf, d_vf, d_wf,
             s_u0, s_v0, s_w0, 
             d_au, d_av, d_aw,
             DWIJ, R2IJ, EPS, XIJ, VIJ):

        # smooth velocities at the ghost points Eq. (23). u0, v0, w0
        # are assumed to be the prescribed wall velocities.
        ug = 2*s_u0[s_idx] - d_uf[d_idx]
        vg = 2*s_v0[s_idx] - d_vf[d_idx]
        wg = 2*s_w0[s_idx] - d_wf[d_idx]
        
        # averaged shear viscosity Eq. (6)
        etaij = self.nu * d_rho[d_idx]

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # inverse mass for destination particle
        mi1 = 1.0/d_m[d_idx]

        # scalar part of the kernel gradient
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
            
        # viscous contribution (third term) from Eq. (8)
        tmp = mi1 * (Vi2 + Vj2) * (etaij * Fij/(R2IJ + EPS))

        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]

class SolidWallPressureBC(Equation):
    """Solid wall pressure boundary condition described in REF1
    Eq. (27)
    
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

    Notes:
    
    For a two fluid system (boundary, fluid), this equation must be
    instantiated with boundary as the destination and fluid as the
    source.

    The boundary particle array must additionally define a property
    :math:`wij` for the denominator in Eq. (27) from REF1. This array
    sums the kernel terms from the ghost particle to the fluid
    particle.
    
    """
    def __init__(self, dest, sources=None, rho0=1.0, p0=100.0,
                 gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=0.0,
                 b=1.0):
        self.rho0 = rho0
        self.p0 = p0
        self.b=b
        self.gx = gx; self.ax = ax
        self.gy = gy; self.ay = ay
        self.gz = gz; self.az = az

        super(SolidWallPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             WIJ, XIJ):

        # numerator of Eq. (27)
        gdotxij = (self.gx-self.ax)*XIJ[0] + \
            (self.gy-self.ay)*XIJ[1] + \
            (self.gz-self.az)*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx] * gdotxij * WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure Eq. (28)
        d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + self.b)
