"""Viscosity functions"""

from pysph.sph.equation import Equation


class LaminarViscosity(Equation):
    def __init__(self, dest, sources, nu, eta=0.01):
        self.nu = nu
        self.eta = eta
        super(LaminarViscosity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho,
             d_au, d_av, d_aw, DWIJ, XIJ, VIJ, R2IJ, HIJ):
        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m[s_idx]

        tmp = mb * 4 * self.nu * Fij/((rhoa + rhob)*(R2IJ + self.eta*HIJ*HIJ))

        # accelerations
        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]


class MonaghanSignalViscosityFluids(Equation):
    def __init__(self, dest, sources, alpha, h):
        self.alpha = 0.125 * alpha * h
        super(MonaghanSignalViscosityFluids, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             d_au, d_av, d_aw, d_cs, s_cs,
             RIJ, HIJ, VIJ, XIJ, DWIJ):

        nua = self.alpha * d_cs[d_idx]
        nub = self.alpha * s_cs[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        mb = s_m[s_idx]

        vabdotrab = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        eta = nua * nub/(nua*rhoa + nub*rhob)
        force = -16 * eta * vabdotrab/(HIJ * (RIJ + 0.01*HIJ*HIJ))

        d_au[d_idx] += -mb * force * DWIJ[0]
        d_av[d_idx] += -mb * force * DWIJ[1]
        d_aw[d_idx] += -mb * force * DWIJ[2]


class ClearyArtificialViscosity(Equation):
    r"""Artificial viscosity proposed By P. Cleary:

    .. math::

       \mathcal{Pi}_{ab} = -\frac{16}{\mu_a \mu_b}{\rho_a \rho_b
       (\mu_a + \mu_b)}\left( \frac{\boldsymbol{v}_{ab} \cdot
       \boldsymbol{r}_{ab}}{\boldsymbol{r}_{ab}^2 + \epsilon} \right),

    where the viscosity is determined from the parameter
    :math:`\alpha` as

    .. math::

        \mu_a = \frac{1}{8}\alpha h_a c_a \rho_a

    This equation is described in the 2005 review paper by Monaghan

    - J. J. Monaghan, "Smoothed Particle Hydrodynamics", Reports on
      Progress in Physics, 2005, 68, pp 1703--1759 [JM05]

    """

    def __init__(self, dest, sources, dim, alpha=1.0):
        self.alpha = alpha
        self.factor = 16.0
        if dim == 3:
            self.factor = 20.0

        # Base class initialization
        super(ClearyArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_rho, s_rho, d_h, s_h, d_cs, s_cs,
             d_au, d_av, d_aw, XIJ, VIJ, R2IJ, EPS, DWIJ):

        # viscosity parameters for each particle Eq. (8.8) in [JM05]
        mua = 0.125 * self.alpha * d_h[d_idx] * d_cs[d_idx] * d_rho[d_idx]
        mub = 0.125 * self.alpha * s_h[s_idx] * s_cs[s_idx] * s_rho[s_idx]

        # \boldsymbol{v}_{ab} \cdot \boldsymbol{r}_{ab}
        dot = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        # Pi_ab term. Eq. (8.9) in [JM05]
        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]
        eta = mua*mub/(rhoa*rhob*(mua + mub))
        piab = -s_m[s_idx] * self.factor*eta * (dot/(R2IJ + EPS))

        # accelerations due to viscosity Eq. (8.2) in [JM05]
        d_au[d_idx] += piab * DWIJ[0]
        d_av[d_idx] += piab * DWIJ[1]
        d_aw[d_idx] += piab * DWIJ[2]


class LaminarViscosityDeltaSPH(Equation):
    r"""See section 2 of the below reference [Sun2017]

    References
    ----------

    .. [Sun2017] P. Sun, A. Colagrossi, S. Marrone, A. Zhang "The $\delta$
       plus-SPH model: simple procedures for a further improvement of the SPH
       scheme", Computer Methods in Applied Mechanics and Engineering 315
       (2017), pp. 25-49.

    """
    def __init__(self, dest, sources, dim, rho0, nu):
        self.dim = dim
        self.rho0 = rho0
        self.nu = nu
        super(LaminarViscosityDeltaSPH, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rho, d_au, d_av, d_aw, HIJ,
             DWIJ, R2IJ, EPS, VIJ, XIJ):
        Vj = s_m[s_idx]/s_rho[s_idx]
        rhoi = d_rho[d_idx]
        vdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        piij = vdotxij/(R2IJ + EPS)

        fac = 2 * (self.dim + 2) * self.nu * self.rho0 * piij * Vj / rhoi
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]
