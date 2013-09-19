"""Basic equations for solid mechanics"""

from pysph.sph.equation import Equation
from textwrap import dedent

class MonaghanArtificialStress(Equation):
    def __init__(self, dest, sources=None, eps=0.3):
        self.eps = eps
        super(MonaghanArtificialStress, self).__init__(dest, sources)

    def cython_code(self):
        code = dedent("""
        cimport cython
        from pysph.sph.solid_mech.linalg cimport get_eigenvalvec
        from pysph.sph.solid_mech.linalg cimport transform2inv
        from pysph.base.point cimport cPoint
        """)
        return dict(helper=code)

    def loop(self, d_idx, d_rho, d_p,
             d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             d_r00, d_r01, d_r02, d_r11, d_r12, d_r22):
        """Compute the stress terms

        Parameters:
        -----------

        d_sxx : DoubleArray
            Stress Tensor Deviatoric components (Symmetric)

        d_rxx : DoubleArray
            Artificial stress components (Symmetric)

        """
        # 1/rho_a^2
        rhoi = d_rho[d_idx]
        rhoi21 = 1./(rhoi * rhoi)

        ## Matrix and vector declarations ##

        # diagonal and off-diaognal terms for the stress tensor
        sd = declare('cPoint')
        ss = declare('cPoint')

        # artificial stress in the principle directions
        rd = declare('cPoint')

        # Matrix of Eigenvectors (columns)
        R = declare('matrix((3,3))')

        # Artificial stress in the original coordinates
        Rab = declare('matrix((3,3))')

        # Eigenvectors
        S = declare('cPoint')

        # get the diagonal terms for the stress tensor adding pressure
        sd.x = d_s00[d_idx] - d_p[d_idx]
        sd.y = d_s11[d_idx] - d_p[d_idx]
        sd.z = d_s22[d_idx] - d_p[d_idx]
        
        ss.x = d_s12[d_idx]
        ss.y = d_s02[d_idx]
        ss.z = d_s01[d_idx]

        # compute the principle stresses
        S = get_eigenvalvec(sd, ss, cython.address(R[0][0]))

        # artificial stress corrections
        if S.x > 0: rd.x = -self.eps * S.x * rhoi21
        else : rd.x = 0

        if S.y > 0: rd.y = -self.eps * S.y * rhoi21
        else : rd.y = 0

        if S.z > 0: rd.z = -self.eps * S.z * rhoi21
        else : rd.z = 0
        
        # transform artificial stresses in original frame
        transform2inv(rd, R, Rab)

        # store the values
        d_r00[d_idx] = Rab[0][0]; d_r11[d_idx] = Rab[1][1]; d_r22[d_idx] = Rab[2][2]
        d_r12[d_idx] = Rab[1][2]; d_r02[d_idx] = Rab[0][2]; d_r01[d_idx] = Rab[0][1]
        
class MomentumEquationWithStress2D(Equation):
    r""" Evaluate the momentum equation:

    :math:`$
    \frac{D\vec{v_a}^i}{Dt} = \sum_b
    m_b\left(\frac{\sigma_a^{ij}}{\rho_a^2} +
    \frac{\sigma_b^{ij}}{\rho_b^2} \right)\nabla_a\,W_{ab}$`

    Artificial stress to remove the tension instability is added to
    the momentum equation as described in `SPH elastic dynamics` by
    J.P. Gray and J.J. Moaghan and R.P. Swift, Computer Methods in
    Applied Mechanical Engineering. vol 190 (2001) pp 6641 - 6662

    """
    def __init__(self, dest, sources=None, wdeltap=-1, n=1):
        self.wdeltap = wdeltap
        self.n = n
        self.with_correction = True
        if wdeltap < 0:
            self.with_correction = False
        super(MomentumEquationWithStress2D, self).__init__(dest, sources)

    def cython_code(self):
        code = dedent("""
        from libc.math cimport pow
        """)
        return dict(helper=code)
        
    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        
    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p,
             d_s00, d_s01, d_s11, s_s00, s_s01, s_s11,
             d_r00, d_r01, d_r11, s_r00, s_r01, s_r11,
             d_au, d_av,
             WIJ=0.0, DWIJ=[0.0, 0.0, 0.0]):

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
    r""" Compute the RHS for the rate of change of stress equation (2D)

    .. raw:: latex

      \[
      \frac{dS^{ij}}{dt} = 2\mu\left(\eps^{ij} -
      \frac{1}{3}\delta^{ij}\eps^{ij}\right) + S^{ik}\Omega^{jk} +
      \Omega^{ik}S^{kj}
      \]

    where,

    :math:`$\eps^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} +
                                  \frac{\partialv^j}{\partialx^i} \right
                                  )$`

    and

    :math:`$\Omega^{ij} = \frac{1}{2}\left( \frac{\partialv^i}{\partialx^j} -
                                  \frac{\partialv^j}{\partialx^i} \right
                                  )$`

    """
    def __init__(self, dest, sources=None, shear_mod=1.0):
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
    def __init__(self, dest, sources=None, alpha=1.0, beta=1.0,
                 eta=0.01):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        super(EnergyEquationWithStress2D,self).__init__(dest, sources)

    def initialize(self, d_idx, d_ae):
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_p, s_p,
             d_cs, s_cs, d_ae,
             XIJ=[0.0, 0.0, 0.0], VIJ=[0.0, 0.0, 0.0],
             DWIJ=[0.0, 0.0, 0.0], HIJ=0.0, R2IJ=0.0, RHOIJ1=0.0):

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
