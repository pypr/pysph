"""Basic equations for solid mechanics"""

from pysph.sph.equation import Equation
from textwrap import dedent

class MonaghanArtificialStress(Equation):
    def __init__(self, dest, sources=None, eps=0.3):
        self.eps = eps
        super(MonaghanArtificialStress, self).__init__(dest, sources)

    def cython_code(self):
        code = dedent("""
        from pysph.sph.smech.linalg cimport _get_eigenvalvec
        from pysph.sph.smech.linalg cimport transform2inv
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
        rho21 = 1./d_rho[d_idx]
        rho21 *= rho21

        ## Matrix and vector declarations ##

        # diagonal and off-diaognal terms for the stress tensor
        sd = declare('matrix((3,))')
        ss = declare('matrix((3,))')

        # artificial stress in the principle directions
        rd = declare('matrix((3,))')

        # Matrix of Eigenvectors (columns)
        R = declare('matrix((3,3))')

        # Artificial stress in the original coordinates
        Rab = declare('matrix((3,3))')

        # Eigenvectors
        S = declare('matrix((3,))')

        # get the diagonal terms for the stress tensor adding pressure
        sd[0] = d_s00[d_idx] - d_p[d_idx]
        sd[1] = d_s11[d_idx] - d_p[d_idx]
        sd[2] = d_s22[d_idx] - d_p[d_idx]
        
        ss[0] = d_s12[d_idx]
        ss[1] = d_s02[d_idx]
        ss[2] = d_s01[d_idx]

        # compute the principle stresses
        _get_eigenvalvec(sd, ss, R, S)

        # correction
        for i in range(3):
            if S[i] > 0:
                rd[i] = -self.eps * S[i] * rho21
            else:
                rd[i] = 0.0
        
        # transform artificial stresses in original frame
        transform2inv(rd, R, Rab)
        
        # store the values
        d_r00[d_idx] = Rab[0][0]; d_r11[d_idx] = Rab[1][1]; d_r22[d_idx] = Rab[2][2]
        d_r12[d_idx] = Rab[1][2]; d_r02[d_idx] = Rab[0][2]; d_r01[d_idx] = Rab[0][1]

        #d_r00[d_idx] = rab_0[0]; d_r11[d_idx] = rab_1[1]; d_r22[d_idx] = rab_2[2]
        #d_r12[d_idx] = rab_2[1]; d_r02[d_idx] = rab_2[0]; d_r01[d_idx] = rab_1[0]
        
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
    def __init__(self, dest, sources=None, wdeltap=1, n=1):
        self.wdeltap = wdeltap
        self.n = n
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

        _XIJ = declare('matrix((3,))')
        
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
        s00a -= pa
        s00b -= pb

        s11a -= pa
        s11b -= pb

        # compute the kernel correction term
        if self.wdeltap > 0:
            fab = WIJ/self.wdeltap
            fab = pow(fab, self.n)
            
            art_stress00 = fab * (r00a + r00b)
            art_stress01 = fab * (r01a + r01b)
            art_stress11 = fab * (r11a + r11b)

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

    
    def loop(self, d_idx, d_s00, d_s01, d_s10, d_s11,
             d_v00, d_v01, d_v10, d_v11,
             d_as00, d_as01, d_as10, d_as11):
        
        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s10 = d_s10[d_idx]
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

        # S_10
        d_as10[d_idx] = d_as01[d_idx]
        
        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
            ( s10*omega10 ) + ( s01*omega10 )
