"""
Generalized Transport Velocity Formulation
##########################################

Some notes on the paper,

- In the viscosity term of equation (17) a factor of '2' is missing.
- A negative sign is missing from equation (22) i.e, either put a negative
  sign in equation (22) or at the integrator step equation(25).
- The Solid Mechanics Equations are not tested.

References
-----------

    .. [ZhangHuAdams2017] Chi Zhang, Xiangyu Y. Hu, Nikolaus A. Adams
       "A generalized transport-velocity formulation for smoothed particle
       hydrodynamics", Journal of Computational Physics 237 (2017),
       pp. 216--232.
"""


from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.scheme import Scheme
from pysph.sph.wc.linalg import mat_vec_mult, mat_mult


def get_particle_array_gtvf(constants=None, **props):
    gtvf_props = [
        'uhat', 'vhat', 'what', 'rho0', 'rhodiv', 'p0', 'auhat', 'avhat',
        'awhat', 'arho', 'arho0'
    ]

    pa = get_particle_array(
        constants=constants, additional_props=gtvf_props, **props
    )
    pa.add_property('gradvhat', stride=9)
    pa.add_property('sigma', stride=9)
    pa.add_property('asigma', stride=9)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'au', 'av', 'aw',
        'pid', 'gid', 'tag'
    ])
    return pa


class GTVFIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.stage1()
        self.do_post_stage(dt, 1)

        self.compute_accelerations(0, update_nnps=False)

        self.stage2()
        # We update domain here alone as positions only change here.
        self.update_domain()
        self.do_post_stage(dt, 2)

        self.compute_accelerations(1)

        self.stage3()
        self.do_post_stage(dt, 3)


class GTVFStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
               d_what, d_auhat, d_avhat, d_awhat, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2*d_awhat[d_idx]

    def stage2(self, d_idx, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_sigma, d_asigma, dt):
        d_rho[d_idx] += dt*d_arho[d_idx]

        i = declare('int')
        for i in range(9):
            d_sigma[d_idx*9 + i] += dt * d_asigma[d_idx*9 + i]

        d_x[d_idx] += dt*d_uhat[d_idx]
        d_y[d_idx] += dt*d_vhat[d_idx]
        d_z[d_idx] += dt*d_what[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]


class ContinuityEquationGTVF(Equation):
    r"""**Evolution of density**

    From [ZhangHuAdams2017], equation (12),

    .. math::
            \frac{\tilde{d} \rho_i}{dt} = \rho_i \sum_j \frac{m_j}{\rho_j}
            \nabla W_{ij} \cdot \tilde{\boldsymbol{v}}_{ij}
    """
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_uhat, d_vhat, d_what,
             s_uhat, s_vhat, s_what, d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij = d_vhat[d_idx] - s_vhat[s_idx]
        whatij = d_what[d_idx] - s_what[s_idx]

        udotdij = DWIJ[0]*uhatij + DWIJ[1]*vhatij + DWIJ[2]*whatij
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


class CorrectDensity(Equation):
    r"""**Density correction**

    From [ZhangHuAdams2017], equation (13),

    .. math::
            \rho_i = \frac{\sum_j m_j W_{ij}}
            {\min(1, \sum_j \frac{m_j}{\rho_j^{*}} W_{ij})}

    where,

    .. math::
            \rho_j^{*} = \text{density before this correction is applied.}
    """
    def initialize(self, d_idx, d_rho, d_rho0, d_rhodiv):
        d_rho0[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0
        d_rhodiv[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_rhodiv, s_m, WIJ, s_rho0):
        d_rho[d_idx] += s_m[s_idx]*WIJ
        d_rhodiv[d_idx] += s_m[s_idx]*WIJ/s_rho0[s_idx]

    def post_loop(self, d_idx, d_rho, d_rhodiv):
        d_rho[d_idx] = d_rho[d_idx] / min(1, d_rhodiv[d_idx])


class MomentumEquationPressureGradient(Equation):
    r"""**Momentum Equation**

    From [ZhangHuAdams2017], equation (17),

    .. math::
            \frac{\tilde{d} \boldsymbol{v}_i}{dt} = - \sum_j m_j \nabla W_{ij}
            \cdot \left[\left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}
            \right)\textbf{I} - \left(\frac{\boldsymbol{A_i}}{\rho_i^2} +
            \frac{\boldsymbol{A_j}}{\rho_j^2} \right)\right] + \sum_j
            \frac{\eta_{ij}\boldsymbol{v}_{ij}}{\rho_i \rho_j r_{ij}}
            \nabla W_{ij} \cdot \boldsymbol{x}_{ij}

    where,

    .. math::
           \boldsymbol{A_{i/j}} = \rho_{i/j} \boldsymbol{v}_{i/j} \otimes
           (\tilde{\boldsymbol{v}}_{i/j} - \boldsymbol{v}_{i/j})

    .. math::
           \eta_{ij} = \frac{2\eta_i \eta_j}{\eta_i + \eta_j}
    .. math::
           \eta_{i/j} = \rho_{i/j} \nu

    for solids, replace :math:`\boldsymbol{A}_{i/j}` with
    :math:`\boldsymbol{\sigma}'_{i/j}`.

    The rate of change of transport velocity is given by,

    .. math::
            (\frac{d\boldsymbol{v}_i}{dt})_c = -p_i^0 \sum_j \frac{m_j}
            {\rho_i^2} \nabla \tilde{W}_{ij}

    where,

    .. math::
            \tilde{W}_{ij} = W(\boldsymbol{x}_ij, \tilde{0.5 h_{ij}})
    .. math::
            p_i^0 = \min(10|p_i|, p_{ref})

    Notes:

    A negative sign in :math:`(\frac{d\boldsymbol{v}_i}{dt})_c` is
    missing in the paper [ZhangHuAdams2017].
    """

    def __init__(self, dest, sources, pref, gx=0.0, gy=0.0, gz=0.0):
        r"""
        Parameters
        ----------
        pref : float
            reference pressure
        gx : float
            body force per unit mass along the x-axis
        gy : float
            body force per unit mass along the y-axis
        gz : float
            body force per unit mass along the z-axis
        """

        self.pref = pref
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat,
                   d_p0, d_p):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

        d_p0[d_idx] = min(10*abs(d_p[d_idx]), self.pref)

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av,
             d_aw, DWIJ, d_p0, d_auhat, d_avhat, d_awhat, XIJ, RIJ, SPH_KERNEL,
             HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        tmp = -d_p0[d_idx] * s_m[s_idx]/rhoi2

        dwijhat = declare('matrix(3)')
        SPH_KERNEL.gradient(XIJ, RIJ, 0.5*HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class MomentumEquationViscosity(Equation):
    r"""**Momentum equation Artificial stress for solids**

    See the class MomentumEquationPressureGradient for details.

    Notes:

    A factor of '2' is missing in the viscosity equation given by
    [ZhangHuAdams2017].
    """
    def __init__(self, dest, sources, nu):
        r"""
        Parameters
        ----------
        nu : float
            viscosity of the fluid.
        """

        self.nu = nu
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au,
             d_av, d_aw, VIJ, R2IJ, EPS, DWIJ, XIJ):
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 4 * (etai * etaj)/(etai + etaj)

        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        tmp = s_m[s_idx]/(d_rho[d_idx] * s_rho[s_idx])
        fac = tmp * etaij * xdotdij/(R2IJ + EPS)

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]


class MomentumEquationArtificialStress(Equation):
    r"""**Momentum equation Artificial stress for solids**

    See the class MomentumEquationPressureGradient for details.
    """
    def __init__(self, dest, sources, dim):
        r"""
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        """
        self.dim = dim
        super(MomentumEquationArtificialStress, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_vec_mult]

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_u, d_v, d_w, d_uhat, d_vhat,
             d_what, s_u, s_v, s_w, s_uhat, s_vhat, s_what, d_au, d_av, d_aw,
             s_m, DWIJ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        i, j = declare('int', 2)
        ui, uj, uidif, ujdif, res = declare('matrix(3)', 5)
        Aij = declare('matrix(9)')

        for i in range(3):
            res[i] = 0.0
            for j in range(3):
                Aij[3*i + j] = 0.0

        ui[0] = d_u[d_idx]
        ui[1] = d_v[d_idx]
        ui[2] = d_w[d_idx]

        uj[0] = s_u[s_idx]
        uj[1] = s_v[s_idx]
        uj[2] = s_w[s_idx]

        uidif[0] = d_uhat[d_idx] - d_u[d_idx]
        uidif[1] = d_vhat[d_idx] - d_v[d_idx]
        uidif[2] = d_what[d_idx] - d_w[d_idx]

        ujdif[0] = s_uhat[s_idx] - s_u[s_idx]
        ujdif[1] = s_vhat[s_idx] - s_v[s_idx]
        ujdif[2] = s_what[s_idx] - s_w[s_idx]

        for i in range(3):
            for j in range(3):
                Aij[3*i + j] = (ui[i]*uidif[j] / rhoi + uj[i]*ujdif[j] / rhoj)

        mat_vec_mult(Aij, DWIJ, 3, res)

        d_au[d_idx] += s_m[s_idx] * res[0]
        d_av[d_idx] += s_m[s_idx] * res[1]
        d_aw[d_idx] += s_m[s_idx] * res[2]


class VelocityGradient(Equation):
    r"""**Gradient of velocity vector**

    .. math::
            (\nabla \otimes \tilde{\boldsymbol{v}})_i = \sum_j \frac{m_j}
            {\rho_j} \tilde{\boldsymbol{v}}_{ij} \otimes \nabla W_{ij}
    """
    def __init__(self, dest, sources, dim):
        r"""
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        """
        self.dim = dim
        super(VelocityGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradvhat):
        for i in range(9):
            d_gradvhat[9*d_idx + i] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_uhat, d_vhat, d_what, s_uhat, s_vhat,
             s_what, s_rho, d_gradvhat, DWIJ):
        i, j = declare('int', 2)
        uhatij = declare('matrix(3)')
        Vj = s_m[s_idx]/s_rho[s_idx]

        uhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        uhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        uhatij[2] = d_what[d_idx] - s_what[s_idx]

        for i in range(3):
            for j in range(3):
                d_gradvhat[d_idx*9 + 3*i + j] += Vj * uhatij[i] * DWIJ[j]


class DeviatoricStressRate(Equation):
    r"""**Stress rate for solids**

    From [ZhangHuAdams2017], equation (5),

    .. math::
            \frac{d \boldsymbol{\sigma}'}{dt} = 2 G (\boldsymbol{\epsilon}
            - \frac{1}{3} \text{Tr}(\boldsymbol{\epsilon})\textbf{I}) +
            \boldsymbol{\sigma}' \cdot \boldsymbol{\Omega}^{T} +
            \boldsymbol{\Omega} \cdot \boldsymbol{\sigma}'

    where,

    .. math::
           \boldsymbol{\Omega_{i/j}} = \frac{1}{2}
           \left(\nabla \otimes \boldsymbol{v}_{i/j} -
           (\nabla \otimes \boldsymbol{v}_{i/j})^{T}\right)

    .. math::
           \boldsymbol{\epsilon_{i/j}} = \frac{1}{2}
           \left(\nabla \otimes \boldsymbol{v}_{i/j} +
           (\nabla \otimes \boldsymbol{v}_{i/j})^{T}\right)

    see the class VelocityGradient for :math:`\nabla \otimes \boldsymbol{v}_i`
   """
    def __init__(self, dest, sources, dim, G):
        r"""
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        G : float
            value of shear modulus
        """
        self.G = G
        self.dim = dim
        super(DeviatoricStressRate, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_vec_mult, mat_mult]

    def initialize(self, d_idx, d_sigma, d_asigma, d_gradvhat):
        i, j, ind = declare('int', 3)
        eps, omega, omegaT, sigmai, dvi = declare('matrix(9)', 5)

        G = self.G

        for i in range(9):
            sigmai[i] = d_sigma[d_idx*9 + i]
            dvi[i] = d_gradvhat[d_idx*9 + i]
            d_asigma[d_idx*9 + i] = 0.0

        eps_trace = 0.0
        for i in range(3):
            for j in range(3):
                eps[3*i + j] = 0.5*(dvi[3*i + j] + dvi[3*j + i])
                omega[3*i + j] = 0.5*(dvi[3*i + j] - dvi[3*j + i])
                if i == j:
                    eps_trace += eps[3*i + j]

        for i in range(3):
            for j in range(3):
                omegaT[3*j + i] = omega[3*i + j]

        smo, oms = declare('matrix(9)', 2)
        mat_mult(sigmai, omegaT, 3, smo)
        mat_mult(omega, sigmai, 3, oms)

        for i in range(3):
            for j in range(3):
                ind = 3*i + j
                d_asigma[d_idx*9 + ind] = 2*G * eps[ind] + smo[ind] + oms[ind]
                if i == j:
                    d_asigma[d_idx*9 + ind] += -2*G * eps_trace/3.0


class MomentumEquationArtificialStressSolid(Equation):
    r"""**Momentum equation Artificial stress for solids**

    See the class MomentumEquationPressureGradient for details.
    """
    def __init__(self, dest, sources, dim):
        r"""
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        """
        self.dim = dim
        super(MomentumEquationArtificialStressSolid, self).__init__(dest,
                                                                    sources)

    def _get_helpers_(self):
        return [mat_vec_mult]

    def loop(self, d_idx, s_idx, d_sigma, s_sigma, d_au, d_av, d_aw, s_m,
             DWIJ):
        i = declare('int')
        sigmaij = declare('matrix(9)')
        res = declare('matrix(3)')

        for i in range(9):
            sigmaij[i] = d_sigma[d_idx*9 + i] + s_sigma[s_idx*9 + i]

        mat_vec_mult(sigmaij, DWIJ, 3, res)

        d_au[d_idx] += s_m[s_idx] * res[0]
        d_av[d_idx] += s_m[s_idx] * res[1]
        d_aw[d_idx] += s_m[s_idx] * res[2]


class GTVFScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, nu, h0, pref,
                 gx=0.0, gy=0.0, gz=0.0, b=1.0, alpha=0.0):
        r"""Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays.
        dim: int
            Dimensionality of the problem.
        rho0: float
            Reference density.
        c0: float
            Reference speed of sound.
        nu: float
            Real viscosity of the fluid.
        h0: float
            Reference smoothing length.
        pref: float
            reference pressure for rate of change of transport velocity.
        gx: float
            Body force acceleration components in x direction.
        gy: float
            Body force acceleration components in y direction.
        gz: float
            Body force acceleration components in z direction.
        b: float
            constant for the equation of state.
        """

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.c0 = c0
        self.nu = nu
        self.h0 = h0
        self.pref = pref
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.b = b
        self.alpha = alpha
        self.solver = None

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """Configure the solver to be generated.

        Parameters
        ----------

        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        from pysph.base.kernels import WendlandQuintic
        if kernel is None:
            kernel = WendlandQuintic(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = GTVFStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        if integrator_cls is not None:
            cls = integrator_cls
            print("Warning: GTVF Integrator is not being used.")
        else:
            cls = GTVFIntegrator
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.wc.transport_velocity import (
            StateEquation, SetWallVelocity, SolidWallPressureBC,
            VolumeSummation, SolidWallNoSlipBC,
            MomentumEquationArtificialViscosity, ContinuitySolid
        )
        all = self.fluids + self.solids

        stage1 = []
        if self.solids:
            eq0 = []
            for solid in self.solids:
                eq0.append(SetWallVelocity(dest=solid, sources=self.fluids))
            stage1.append(Group(equations=eq0, real=False))

        eq1 = []
        for fluid in self.fluids:
            eq1.append(ContinuityEquationGTVF(dest=fluid, sources=self.fluids))
            if self.solids:
                eq1.append(
                    ContinuitySolid(dest=fluid, sources=self.solids)
                )
        stage1.append(Group(equations=eq1, real=False))

        eq2, stage2 = [], []
        for fluid in self.fluids:
            eq2.append(CorrectDensity(dest=fluid, sources=all))
        stage2.append(Group(equations=eq2, real=False))

        eq3 = []
        for fluid in self.fluids:
            eq3.append(
                StateEquation(dest=fluid, sources=None, p0=self.pref,
                              rho0=self.rho0, b=1.0)
            )
        stage2.append(Group(equations=eq3, real=False))

        g2_s = []
        for solid in self.solids:
            g2_s.append(VolumeSummation(dest=solid, sources=all))
            g2_s.append(SolidWallPressureBC(
                dest=solid, sources=self.fluids, b=1.0, rho0=self.rho0,
                p0=self.pref, gx=self.gx, gy=self.gy, gz=self.gz
            ))
        if g2_s:
            stage2.append(Group(equations=g2_s, real=False))

        eq4 = []
        for fluid in self.fluids:
            eq4.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, pref=self.pref,
                    gx=self.gx, gy=self.gy, gz=self.gz
                ))
            if self.alpha > 0.0:
                eq4.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    ))
            if self.nu > 0.0:
                eq4.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=all, nu=self.nu
                    ))
                if self.solids:
                    eq4.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.solids, nu=self.nu
                        ))
            eq4.append(
                MomentumEquationArtificialStress(
                    dest=fluid, sources=self.fluids, dim=self.dim
                ))
        stage2.append(Group(equations=eq4, real=True))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_gtvf(name='junk')
        props = list(dummy.properties.keys())
        props += [dict(name=p, stride=v) for p, v in dummy.stride.items()]
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)

        solid_props = ['uf', 'vf', 'wf', 'vg', 'ug', 'wij', 'wg', 'V']
        props += solid_props
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
