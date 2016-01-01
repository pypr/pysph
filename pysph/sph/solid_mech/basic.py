"""
Basic Equations for Solid Mechanics
###################################

References
----------
.. [Gray2001] J. P. Gray et al., "SPH elastic dynamics", Computer Methods 
    in Applied Mechanics and Engineering, 190 (2001), pp 6641 - 6662.
"""

from pysph.sph.equation import Equation
from textwrap import dedent

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

    def loop(self, d_idx, d_rho, d_p,
             d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
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
        rhoi21 = 1./(rhoi * rhoi)

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

        S[1][2] = S[2][1] = d_s12[d_idx]
        S[0][2] = S[2][0] = d_s02[d_idx]
        S[0][1] = S[1][0] = d_s01[d_idx]

        # compute the principle stresses
        eigen_decomposition(S, R, cython.address(V[0]))

        # artificial stress corrections
        if V[0] > 0: rd[0] = -self.eps * V[0] * rhoi21
        else : rd[0] = 0

        if V[1] > 0: rd[1] = -self.eps * V[1] * rhoi21
        else : rd[1] = 0

        if V[2] > 0: rd[2] = -self.eps * V[2] * rhoi21
        else : rd[2] = 0

        # transform artificial stresses in original frame
        transform_diag_inv(cython.address(rd[0]), R, Rab)

        # store the values
        d_r00[d_idx] = Rab[0][0]; d_r11[d_idx] = Rab[1][1]; d_r22[d_idx] = Rab[2][2]
        d_r12[d_idx] = Rab[1][2]; d_r02[d_idx] = Rab[0][2]; d_r01[d_idx] = Rab[0][1]

class MomentumEquationWithStress2D(Equation):
    r"""**Momentum Equation with Artificial Stress**
    
    .. math::
    
        \frac{D\vec{v_a}^i}{Dt} = \sum_b m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2} 
        +\frac{\sigma_b^{ij}}{\rho_b^2} + R_{ab}^{ij}f^n \right)\nabla_a W_{ab}
    
    where
    
    .. math::
    
        f_{ab} = \frac{W(r_{ab})}{W(\Delta p)}\\
        
        R_{ab}^{ij} = R_{a}^{ij} + R_{b}^{ij}
    """
    def __init__(self, dest, sources, wdeltap=-1, n=1):
        r"""
        Parameters
        ----------
        wdeltap : float
            evaluated value of :math:`W(\Delta p)`
        n : float
            constant
        with_correction : bool
            switch for using tensile instability correction
        """
        
        self.wdeltap = wdeltap
        self.n = n
        self.with_correction = True
        if wdeltap < 0:
            self.with_correction = False
        super(MomentumEquationWithStress2D, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p,
             d_s00, d_s01, d_s11, s_s00, s_s01, s_s11,
             d_r00, d_r01, d_r11, s_r00, s_r01, s_r11,
             d_au, d_av, WIJ, DWIJ):

        pa = d_p[d_idx]
        pb = s_p[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        rhoa21 = 1./(rhoa * rhoa)
        rhob21 = 1./(rhob * rhob)

        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]

        s00b = s_s00[s_idx]
        s01b = s_s01[s_idx]
        s10b = s_s01[s_idx]
        s11b = s_s11[s_idx]

        r00a = d_r00[d_idx]
        r01a = d_r01[d_idx]
        r10a = d_r01[d_idx]
        r11a = d_r11[d_idx]

        r00b = s_r00[s_idx]
        r01b = s_r01[s_idx]
        r10b = s_r01[s_idx]
        r11b = s_r11[s_idx]

        # Add pressure to the deviatoric components
        s00a = s00a - pa
        s00b = s00b - pb

        s11a = s11a - pa
        s11b = s11b - pb

        # compute the kernel correction term
        if self.with_correction:
            fab = WIJ/self.wdeltap
            fab = pow(fab, self.n)

            art_stress00 = fab * (r00a + r00b)
            art_stress01 = fab * (r01a + r01b)
            art_stress11 = fab * (r11a + r11b)
        else:
            art_stress00 = 0.0
            art_stress01 = 0.0
            art_stress11 = 0.0

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += mb * (s00a*rhoa21 + s00b*rhob21 + art_stress00) * DWIJ[0] + \
            mb * (s01a*rhoa21 + s01b*rhob21 + art_stress01) * DWIJ[1]

        d_av[d_idx] += mb * (s10a*rhoa21 + s10b*rhob21 + art_stress01) * DWIJ[0] + \
            mb * (s11a*rhoa21 + s11b*rhob21 + art_stress11) * DWIJ[1]

class HookesDeviatoricStressRate2D(Equation):
    r""" **Rate of change of stress (2D)**

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
    def __init__(self, dest, sources, shear_mod):
        r"""
        Parameters
        ----------
        shear_mod : float
            shear modulus (:math:`\mu`)
        """
        
        self.shear_mod = shear_mod
        super(HookesDeviatoricStressRate2D, self).__init__(dest, sources)

    def initialize(self, d_idx, d_as00, d_as01, d_as11):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as11[d_idx] = 0.0

    def loop(self, d_idx, d_s00, d_s01, d_s11,
             d_v00, d_v01, d_v10, d_v11,
             d_as00, d_as01, d_as11):

        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]

        # strain rate tensor is symmetric
        eps00 = v00
        eps01 = 0.5 * (v01 + v10)

        eps10 = eps01
        eps11 = v11

        # rotation tensor is asymmetric
        omega01 = 0.5 * (v01 - v10)
        omega10 = -omega01

        tmp = 2.0*self.shear_mod
        trace = 1.0/3.0 * (eps00 + eps11)

        # S_00
        d_as00[d_idx] = tmp*( eps00 - trace ) + \
            ( s01*omega01 ) + ( s10*omega01 )

        # S_01
        d_as01[d_idx] = tmp*(eps01) + \
            ( s00*omega10 ) + ( s11*omega01 )

        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
            ( s10*omega10 ) + ( s01*omega10 )

class EnergyEquationWithStress2D(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0,
                 eta=0.01):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        super(EnergyEquationWithStress2D,self).__init__(dest, sources)

    def initialize(self, d_idx, d_ae):
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p,
             d_cs, s_cs, d_ae, XIJ, VIJ, DWIJ, HIJ, R2IJ, RHOIJ1):

        rhoa = d_rho[d_idx]
        ca = d_cs[d_idx]
        pa = d_p[d_idx]

        rhob = s_rho[s_idx]
        cb = s_cs[s_idx]
        pb = s_p[s_idx]
        mb = s_m[s_idx]

        rhoa2 = 1./(rhoa*rhoa)
        rhob2 = 1./(rhob*rhob)

        # artificial viscosity
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + self.eta*self.eta*HIJ*HIJ)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1


        vijdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]

        # thermal energy contribution
        d_ae[d_idx] += 0.5 * mb * (pa*rhoa2 + pb*rhob2 + piij)

    def post_loop(self, d_idx, d_rho,
                  d_s00, d_s01, d_s11, s_s00, s_s01, s_s11,
                  d_v00, d_v01, d_v10, d_v11,
                  d_ae):

        # particle density
        rhoa = d_rho[d_idx]

        # deviatoric stress rate (symmetric)
        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]

        # strain rate tensor (symmetric)
        eps00 = d_v00[d_idx]
        eps01 = 0.5 * (d_v01[d_idx] + d_v10[d_idx])

        eps10 = eps01
        eps11 = d_v11[d_idx]

        # energy acclerations
        sdoteij = s00a*eps00 +  s01a*eps01 + s10a*eps10 + s11a*eps11
        d_ae[d_idx] += 1./rhoa * sdoteij
