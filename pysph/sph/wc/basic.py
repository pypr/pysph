"""
Basic WCSPH Equations
#####################
"""

from pysph.sph.equation import Equation
from textwrap import dedent

class TaitEOS(Equation):
    r"""**Tait equation of state for water-like fluids**

    :math:`p_a = \frac{c_{0}^2\rho_0}{\gamma}\left(
    \left(\frac{\rho_a}{\rho_0}\right)^{\gamma} -1\right)`
    
    References
    ----------
    .. [Cole1948] H. R. Cole, "Underwater Explosions", Princeton University
        Press, 1948.
    
    .. [Batchelor2002] G. Batchelor, "An Introduction to Fluid Dynamics", 
        Cambridge University Press, 2002.
    
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics", 
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759. 

    """
    def __init__(self, dest, sources, rho0, c0, gamma, p0=0.0):
                     
        r"""
        Parameters
        ----------
        rho0 : float
            reference density of fluid particles
        c0 : float
            maximum speed of sound expected in the system
        gamma : float
            constant
        p0 : float
            reference pressure in the system (defaults to zero).
        
        Notes
        -----
        The reference speed of sound, c0, is to be taken approximately as
        10 times the maximum expected velocity in the system. The particle
        sound speed is given by the usual expression:
    
        :math:`c_a = \sqrt{\frac{\partial p}{\partial \rho}}`

        """
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.B = rho0*c0*c0/gamma
        self.p0 = p0
        
        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.p0 + self.B * (tmp - 1.0)
        d_cs[d_idx] = self.c0 * pow( ratio, self.gamma1 )

class TaitEOSHGCorrection(Equation):
    r"""**Tait Equation of State with Hughes and Graham Correction**

    .. math::
        p_a = \frac{c_{0}^2\rho_0}{\gamma}\left(
        \left(\frac{\rho_a}{\rho_0}\right)^{\gamma} -1\right)
    
    where
    
    .. math::
    
        \rho_{a}=\begin{cases}\rho_{a} & \rho_{a}\geq\rho_{0}\\
        \rho_{0} & \rho_{a}<\rho_{0}\end{cases}`

    References
    ----------
    .. [Hughes2010] J. P. Hughes and D. I. Graham, "Comparison of incompressible 
        and weakly-compressible SPH models for free-surface water flows",
        Journal of Hydraulic Research, 48 (2010), pp. 105-117.

    """
    
    def __init__(self, dest, sources, rho0, c0, gamma):
        
        r"""
        Parameters
        ----------
        rho0 : float
            reference density
        c0 : float
            reference speed of sound
        gamma : float
            constant
            
        Notes
        -----
        The correction is to be applied on boundary particles and imposes
        a minimum value of the density (rho0) which is set upon
        instantiation. This correction avoids particle sticking behaviour
        at walls.
        """
        
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.B = rho0*c0*c0/gamma
        super(TaitEOSHGCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs):
        if d_rho[d_idx] < self.rho0:
            d_rho[d_idx] = self.rho0

        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.B * (tmp - 1.0)
        d_cs[d_idx] = self.c0 * pow( ratio, self.gamma1 )

class MomentumEquation(Equation):
    r"""**Classic Monaghan Style Momentum Equation with Artificial Viscosity**
    
    .. math::
    
        \frac{d\mathbf{v}_{a}}{dt}=-\sum_{b}m_{b}\left(\frac{p_{b}}
        {\rho_{b}^{2}}+\frac{p_{a}}{\rho_{a}^{2}}+\Pi_{ab}\right)
        \nabla_{a}W_{ab}
        
    where
    
    .. math::
    
        \Pi_{ab}=\begin{cases}
        \frac{-\alpha\bar{c}_{ab}\mu_{ab}+\beta\mu_{ab}^{2}}{\bar{\rho}_{ab}} & 
        \mathbf{v}_{ab}\cdot\mathbf{r}_{ab}<0;\\
        0 & \mathbf{v}_{ab}\cdot\mathbf{r}_{ab}\geq0;
        \end{cases}
        
    with
    
    .. math::
    
        \mu_{ab}=\frac{h\mathbf{v}_{ab}\cdot\mathbf{r}_{ab}}
        {\mathbf{r}_{ab}^{2}+\eta^{2}}\\
        
        \bar{c}_{ab} = \frac{c_a + c_b}{2}\\
        
        \bar{\rho}_{ab} = \frac{\rho_a + \rho_b}{2}
        
    References
    ----------
    .. [Monaghan1992] J. Monaghan, Smoothed Particle Hydrodynamics, "Annual 
        Review of Astronomy and Astrophysics", 30 (1992), pp. 543-574.
    """
    def __init__(self, dest, sources, c0,
                 alpha=1.0, beta=1.0, gx=0.0, gy=0.0, gz=0.0,
                 tensile_correction=False):
        
        r"""
        Parameters
        ----------
        c0 : float
            reference speed of sound
        alpha : float
            produces a shear and bulk viscosity
        beta : float
            used to handle high Mach number shocks
        gx : float
            body force per unit mass along the x-axis
        gy : float
            body force per unit mass along the y-axis
        gz : float
            body force per unit mass along the z-axis
        tensilte_correction : bool
            switch for tensile instability correction (Default: False)
        """
        
        self.alpha = alpha
        self.beta = beta
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0

        self.tensile_correction = tensile_correction

        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, DT_ADAPT, WIJ, WDP):

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        # compute the CFL time step factor
        _dt_cfl = 0.0
        if R2IJ > 1e-12:
            _dt_cfl = abs( HIJ * vijdotxij/R2IJ ) + self.c0
            DT_ADAPT[0] = max(_dt_cfl, DT_ADAPT[0])

        tmpi = d_p[d_idx]*rhoi21
        tmpj = s_p[s_idx]*rhoj21

        fij = WIJ/WDP
        Ri = 0.0; Rj = 0.0

        #tmp = d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21
        #tmp = tmpi + tmpj

        # tensile instability correction
        if self.tensile_correction:
            fij = fij*fij
            fij = fij*fij

            if d_p[d_idx] > 0 :
                Ri = 0.01 * tmpi
            else:
                Ri = 0.2*abs( tmpi )

            if s_p[s_idx] > 0:
                Rj = 0.01 * tmpj
            else:
                Rj = 0.2 * abs( tmpj )

        # gradient and correction terms
        tmp = (tmpi + tmpj) + (Ri + Rj)*fij

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, DT_ADAPT):
        d_au[d_idx] +=  self.gx
        d_av[d_idx] +=  self.gy
        d_aw[d_idx] +=  self.gz

        acc2 = ( d_au[d_idx]*d_au[d_idx] + \
                    d_av[d_idx]*d_av[d_idx] + \
                    d_aw[d_idx]*d_aw[d_idx] )

        # store the square of the max acceleration
        DT_ADAPT[1] = max( acc2, DT_ADAPT[1] )

class MomentumEquationDeltaSPH(Equation):
    r"""**Momentum equation defined in JOSEPHINE and the delta-SPH model**

    .. math::
    
        \frac{du_{i}}{dt}=-\frac{1}{\rho_{i}}\sum_{j}\left(p_{j}+p_{i}\right)
        \nabla_{i}W_{ij}V_{j}+\mathbf{g}_{i}+\alpha hc_{0}\rho_{0}\sum_{j}
        \pi_{ij}\nabla_{i}W_{ij}V_{j}
    
    where
    
    .. math:: 
    
        \pi_{ij}=\frac{\mathbf{u}_{ij}\cdot\mathbf{r}_{ij}}
        {|\mathbf{r}_{ij}|^{2}}
    
    References
    ----------
    .. [Marrone2011] S. Marrone et al., "delta-SPH model for simulating 
        violent impact flows", Computer Methods in Applied Mechanics and 
        Engineering, 200 (2011), pp 1526--1542.

    .. [Cherfils2012] J. M. Cherfils et al., "JOSEPHINE: A parallel SPH code for 
        free-surface flows", Computer Physics Communications, 183 (2012), 
        pp 1468--1480.
       
    """
    def __init__(
        self, dest, sources, rho0, c0, alpha=1.0,
        gx=0.0, gy=0.0, gz=0.0):

        r"""
        Parameters
        ----------
        rho0 : float
            reference density
        c0 : float
            reference speed of sound
        alpha : float
            coefficient used to control the intensity of the 
            diffusion of velocity
        gx : float
            body force per unit mass along the x-axis
        gy : float
            body force per unit mass along the y-axis
        gz : float
            body force per unit mass along the z-axis
        
        Notes
        -----
        Artificial viscosity is used in this momentum equation and is
        controlled by the parameter :math:`\alpha`. This form of the
        artificial viscosity is similar but not identical to the
        Monaghan-style artificial viscosity.
        """
        
        self.alpha = alpha
        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.c0 = c0
        self.rho0 = rho0

        super(MomentumEquationDeltaSPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, WIJ, DWIJ):

        # src paricle volume mj/rhoj
        Vj = s_m[s_idx]/s_rho[s_idx]

        pi = d_p[d_idx]
        pj = s_p[s_idx]

        # viscous contribution second part of eqn (5b) in [Maronne2011]
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        piij = self.alpha * HIJ * self.c0 * self.rho0 * vijdotxij/(R2IJ + EPS)

        # gradient and viscous terms eqn 5b in [Maronne2011]
        tmp = -Vj/d_rho[d_idx] * (pi + pj) + piij * Vj/d_rho[d_idx]

        # accelerations
        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, DT_ADAPT):
        d_au[d_idx] +=  self.gx
        d_av[d_idx] +=  self.gy
        d_aw[d_idx] +=  self.gz

        acc2 = ( d_au[d_idx]*d_au[d_idx] + \
                    d_av[d_idx]*d_av[d_idx] + \
                    d_aw[d_idx]*d_aw[d_idx] )

        # store the square of the max acceleration
        DT_ADAPT[1] = max( acc2, DT_ADAPT[1] )

class ContinuityEquationDeltaSPH(Equation):
    r"""**Continuity equation with dissipative terms**

    :math:`\frac{d\rho_a}{dt} = \sum_b \rho_a \frac{m_b}{\rho_b}
    \left( \boldsymbol{v}_{ab}\cdot \nabla_a W_{ab} + \delta \eta_{ab}
    \cdot \nabla_{a} W_{ab} (h_{ab}\frac{c_{ab}}{\rho_a}(\rho_b -
    \rho_a)) \right)`

    References
    ----------
    .. [Marrone2011] S. Marrone et al., "delta-SPH model for simulating 
        violent impact flows", Computer Methods in Applied Mechanics and 
        Engineering, 200 (2011), pp 1526--1542.
    """
    def __init__(self, dest, sources, c0, delta=0.1):
        r"""
        Parameters
        ----------
        c0 : float
            reference speed of sound
        delta : float
            coefficient used to control the intensity of diffusion of density
        """
        self.c0 = c0
        self.delta = delta
        super(ContinuityEquationDeltaSPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, d_cs, s_cs, d_rho, s_rho,
             DWIJ, VIJ, XIJ, RIJ, HIJ, EPS):

        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        Vj = s_m[s_idx]/rhoj

        # v_{ij} \cdot \nabla W
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]

        # eta_{ij} \cdot \nabla W
        etadotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        etadotdwij /= (RIJ + EPS)

        # celerity (sound speed)
        #cij =  max( d_cs[d_idx], s_cs[s_idx] )
        cij = self.c0
        psi_ij = self.delta * HIJ * cij * (rhoj - rhoi)

        # standard term with dissipative penalization eqn (5a)
        d_arho[d_idx] += rhoi*vijdotdwij*Vj + psi_ij*etadotdwij*Vj

class UpdateSmoothingLengthFerrari(Equation):
    r"""**Update the particle smoothing lengths**

    :math:`h_a = hdx \left(\frac{m_a}{\rho_a}\right)^{\frac{1}{d}}`

    References
    ----------
    .. [Ferrari2009] A. Ferrari et al., "A new 3D parallel SPH scheme for free
        surface flows", Computers and Fluids, 38 (2009), pp. 1203--1217.
    """
    
    def __init__(self, dest, sources, dim, hdx):
        r"""
        Parameters
        ----------
        dim : float
            number of dimensions
        hdx : float
            scaling factor
            
        Notes
        -----
        Ideally, the kernel scaling factor should be determined from the
        kernel used based on a linear stability analysis. The default
        value of (hdx=1) reduces to the formulation suggested by Ferrari
        et al. who used a Cubic Spline kernel.
    
        Typically, a change in the smoothing length should mean the
        neighbors are re-computed which in PySPH means the NNPS must be
        updated. This equation should therefore be placed as the last
        equation so that after the final corrector stage, the smoothing
        lengths are updated and the new NNPS data structure is computed.
    
        Note however that since this is to be used with incompressible flow
        equations, the density variations are small and hence the smoothing
        lengths should also not vary too much.
        """
        self.dim1 = 1./dim
        self.hdx = hdx

        super(UpdateSmoothingLengthFerrari, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_h, d_m):
        # naive estimate of particle volume
        Vj = d_m[d_idx]/d_rho[d_idx]

        d_h[d_idx] = self.hdx * pow(Vj, self.dim1)


class PressureGradientUsingNumberDensity(Equation):
    r"""Pressure gradient discretized using number density:
    
    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = -\frac{1}{m_a}\sum_b
        (\frac{p_a}{V_a^2} + \frac{p_b}{V_b^2})\nabla_a W_{ab}

    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        
    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, 
             d_au, d_av, d_aw, d_p, s_p, d_V, s_V, DWIJ):

        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # pressure gradient term
        pi = d_p[d_idx]; pj = s_p[s_idx]
        pij = pi*Vi2 + pj*Vj2

        # accelerations
        tmp = -pij * 1.0/(d_m[d_idx])

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]
