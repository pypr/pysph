"""Basic SPH equations"""

from pysph.sph.equation import Equation

class SummationDensity(Equation):
    r"""Goold old Summation density:

    :math:`$\rho_a = \sum_b m_b W_{ab}$`

    """
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ

class BodyForce(Equation):
    def __init__(self, dest, sources,
                 fx=0.0, fy=0.0, fz=0.0):
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

    :math:`$\frac{\partial v^i}{\partial x^j} = \sum_{b}\frac{m_b}{\rho_b}(v_b
    - v_a)\frac{\partial W_{ab}}{\partial x_a^j}$`

    The tensor properties are stored in the variables v_ij where 'i'
    refers to the velocity component and 'j' refers to the spatial
    component. Thus v_21 is

    :math:`$\frac{\partial w}{\partial y}$`

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

    :math:`$p = c_0^2(\rho_0 - rho)$`

    """
    def __init__(self, dest, sources=None,
                 rho0=1000.0, c0=1.0):
        self.rho0 = rho0
        self.c0 = c0
        self.c02 = c0 * c0
        super(IsothermalEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        d_p[d_idx] = self.c02 * (d_rho[d_idx] - self.rho0)

class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`$\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab} $`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class MonaghanArtificialViscosity(Equation):
    def __init__(self, dest, sources=None, alpha=1.0, beta=1.0, eta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, DWIJ, DT_ADAPT):

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + self.eta*self.eta*HIJ*HIJ)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]

class XSPHCorrection(Equation):
    def __init__(self, dest, sources=None, eps=0.5):
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
