"""
EDAC SPH formulation
#####################

Equations for the Entropically Damped Artificial Compressibility SPH scheme.

Please note that this scheme is still under development and this module may
change at some point in the future.

References
----------

    .. [PRKP2017] Prabhu Ramachandran and Kunal Puri, Entropically damped
       artificial compressibility for SPH, under review, 2017.
       http://arxiv.org/pdf/1311.2167v2.pdf

"""

from math import sin
from math import pi as M_PI

from pysph.base.utils import get_particle_array
from pysph.base.utils import DEFAULT_PROPS
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.scheme import Scheme, add_bool_argument


EDAC_PROPS = ('ap', 'au', 'av', 'aw', 'ax', 'ay', 'az',
              'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'p0', 'V')


def get_particle_array_edac(constants=None, **props):
    "Get the fluid array for the transport velocity formulation"

    pa = get_particle_array(
        constants=constants, additional_props=EDAC_PROPS, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                          'au', 'av', 'aw', 'ap', 'm', 'h'])

    return pa


EDAC_SOLID_PROPS = ('ap', 'p0', 'wij', 'uf', 'vf', 'wf', 'ug', 'vg', 'wg',
                    'ax', 'ay', 'az', 'V')


def get_particle_array_edac_solid(constants=None, **props):
    "Get the fluid array for the transport velocity formulation"

    pa = get_particle_array(
        constants=constants, additional_props=EDAC_SOLID_PROPS, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                          'h'])

    return pa


class ComputeAveragePressure(Equation):
    """Simple function to compute the average pressure at each particle.

    This is used for the Basa, Quinlan and Lastiwka correction from their 2009
    paper.  This equation should be in a separate group and computed before the
    Momentum equation.
    """
    def initialize(self, d_idx, d_pavg, d_nnbr):
        d_pavg[d_idx] = 0.0
        d_nnbr[d_idx] = 0.0

    def loop(self, d_idx, d_pavg, s_idx, s_p, d_nnbr):
        d_pavg[d_idx] += s_p[s_idx]
        d_nnbr[d_idx] += 1.0

    def post_loop(self, d_idx, d_pavg, d_nnbr):
        if d_nnbr[d_idx] > 0:
            d_pavg[d_idx] /= d_nnbr[d_idx]


class EDACStep(IntegratorStep):
    """Standard Predictor Corrector integrator for the WCSPH formulation

    Use this integrator for WCSPH formulations. In the predictor step,
    the particles are advanced to `t + dt/2`. The particles are then
    advanced with the new force computed at this position.

    This integrator can be used in PEC or EPEC mode.

    The same integrator can be used for other problems. Like for
    example solid mechanics (see SolidMechStep)

    """
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_ax, d_ay, d_az, d_ap, dt=0.0):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_ax, d_ay, d_az, d_ap, dt=0.0):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


class SolidWallPressureBC(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(SolidWallPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class ClampWallPressure(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def post_loop(self, d_idx, d_p):
        if d_p[d_idx] < 0.0:
            d_p[d_idx] = 0.0


class SourceNumberDensity(Equation):
    r"""Evaluates the number density due to the source particles"""
    def initialize(self, d_idx, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, WIJ):
        d_wij[d_idx] += WIJ


class SetWallVelocity(Equation):
    r"""Extrapolating the fluid velocity on to the wall Eq. (22) in REF1:

    .. math::

        \tilde{\boldsymbol{v}}_a = \frac{\sum_b\boldsymbol{v}_b W_{ab}}
        {\sum_b W_{ab}}

    Notes:

    This should be used only after (or in the same group) as the
    SolidWallPressureBC equation.

    The destination particle array for this equation should define the
    *filtered* velocity variables :math:`uf, vf, wf`.

    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, WIJ):
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

        # Note that d_wij is already computed for the pressure BC.
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]


class NoSlipVelocityExtrapolation(Equation):
    '''No Slip boundary condition on the wall

    The velocity of the fluid is extrapolated over to the wall using
    shepard extrapolation. The velocity normal to the wall is reflected back
    to impose no penetration.
    '''
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, s_u, s_v, s_w, WIJ,
             XIJ):
        d_u[d_idx] += s_u[s_idx]*WIJ
        d_v[d_idx] += s_v[s_idx]*WIJ
        d_w[d_idx] += s_w[s_idx]*WIJ

    def post_loop(self, d_idx, d_wij, d_u, d_v, d_w, d_xn, d_yn, d_zn):
        if d_wij[d_idx] > 1e-14:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]
            d_w[d_idx] /= d_wij[d_idx]

        projection = d_u[d_idx]*d_xn[d_idx] +\
            d_v[d_idx]*d_yn[d_idx] + d_w[d_idx]*d_zn[d_idx]

        d_u[d_idx] = d_u[d_idx] - 2 * projection * d_xn[d_idx]
        d_v[d_idx] = d_v[d_idx] - 2 * projection * d_yn[d_idx]
        d_w[d_idx] = d_w[d_idx] - 2 * projection * d_zn[d_idx]


class NoSlipAdvVelocityExtrapolation(Equation):
    '''No Slip boundary condition on the wall

    The advection velocity of the fluid is extrapolated over to the wall
    using shepard extrapolation. The advection velocity normal to the wall
    is reflected back to impose no penetration.
    '''
    def initialize(self, d_idx, d_uhat, d_vhat, d_what):
        d_uhat[d_idx] = 0.0
        d_vhat[d_idx] = 0.0
        d_what[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uhat, d_vhat, d_what, s_uhat, s_vhat,
             s_what, WIJ, XIJ):
        d_uhat[d_idx] += s_uhat[s_idx]*WIJ
        d_vhat[d_idx] += s_vhat[s_idx]*WIJ
        d_what[d_idx] += s_what[s_idx]*WIJ

    def post_loop(self, d_idx, d_wij, d_uhat, d_vhat, d_what, d_xn, d_yn,
                  d_zn):
        if d_wij[d_idx] > 1e-14:
            d_uhat[d_idx] /= d_wij[d_idx]
            d_vhat[d_idx] /= d_wij[d_idx]
            d_what[d_idx] /= d_wij[d_idx]

        projection = d_uhat[d_idx]*d_xn[d_idx] +\
            d_vhat[d_idx]*d_yn[d_idx] + d_what[d_idx]*d_zn[d_idx]

        d_uhat[d_idx] = d_uhat[d_idx] - 2 * projection * d_xn[d_idx]
        d_vhat[d_idx] = d_vhat[d_idx] - 2 * projection * d_yn[d_idx]
        d_what[d_idx] = d_what[d_idx] - 2 * projection * d_zn[d_idx]


class MomentumEquation(Equation):
    r"""Momentum equation (gradient of pressure) based on the number
    density formulation of Hu and Adams JCP 213 (2006), 844-861.

    """
    def __init__(self, dest, sources, c0, gx=0.0, gy=0.0, gz=0.0, tdamp=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0
        self.tdamp = tdamp

        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, d_p, d_V, d_au, d_av, d_aw,
             s_m, s_rho, s_p, s_V, DWIJ):

        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        p_i = d_p[d_idx]
        p_j = s_p[s_idx]

        pij = rhoj * p_i + rhoi * p_j
        pij /= (rhoj + rhoi)

        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # inverse mass of destination particle
        mi1 = 1.0/d_m[d_idx]

        tmp = -pij * mi1 * (Vi2 + Vj2)

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, t):
        damping_factor = 1.0
        if t < self.tdamp:
            damping_factor = 0.5 * (sin((-0.5 + t/self.tdamp)*M_PI) + 1.0)
        d_au[d_idx] += damping_factor*self.gx
        d_av[d_idx] += damping_factor*self.gy
        d_aw[d_idx] += damping_factor*self.gz


class EDACEquation(Equation):
    def __init__(self, dest, sources, cs, nu, rho0):
        self.cs = cs
        self.nu = nu
        self.rho0 = rho0

        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, d_V, s_idx, s_m, s_rho, s_p,
             s_V, DWIJ, VIJ, XIJ, R2IJ, EPS):

        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        etai = d_rho[d_idx]
        etaj = s_rho[s_idx]
        etaij = 2 * self.nu * (etai * etaj)/(etai + etaj)

        # This is the same as continuity acceleration times cs^2
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += rhoi/rhoj*self.cs*self.cs*s_m[s_idx]*vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1.0/d_m[d_idx]*(Vi2 + Vj2)*etaij*xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p[s_idx])


class MomentumEquationPressureGradient(Equation):

    r"""Momentum equation for internal flows with EDAC.

    This uses the basic formulation from the TVF scheme but modifies it to
    subtract an average pressure from Basa, Quinlan and Lastiwka correction
    from their 2009 paper.
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

        The body forces are damped according to Eq. (13) in [Adami2012] to
        avoid instantaneous accelerations. By default, damping is neglected.
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
             d_au, d_av, d_aw, d_p, d_pavg, s_p,
             d_auhat, d_avhat, d_awhat, d_V, s_V, DWIJ):

        # averaged pressure Eq. (7)
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        pavg = d_pavg[d_idx]
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        pij = rhoj * (pi - pavg) + rhoi * (pj - pavg)
        pij /= (rhoj + rhoi)

        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

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
            damping_factor = 0.5 * (sin((-0.5 + t/self.tdamp)*M_PI) + 1.0)

        d_au[d_idx] += self.gx * damping_factor
        d_av[d_idx] += self.gy * damping_factor
        d_aw[d_idx] += self.gz * damping_factor


class EDACTVFStep(IntegratorStep):

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au,
               d_av, d_auhat, d_avhat, d_awhat, d_uhat, d_vhat, d_what,
               d_aw, d_ap, dt=0.0):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2*d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_what[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_auhat, d_avhat, d_awhat, d_uhat, d_vhat, d_what,
               d_ap, dt=0.0):
        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt*d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


class EDACScheme(Scheme):
    def __init__(self, fluids, solids, dim, c0, nu, rho0, pb=0.0,
                 gx=0.0, gy=0.0, gz=0.0, tdamp=0.0, eps=0.0, h=0.0,
                 edac_alpha=0.5, alpha=0.0, bql=True, clamp_p=False,
                 inlet_outlet_manager=None, inviscid_solids=None):
        """The EDAC scheme.

        Parameters
        ----------

        fluids : list(str)
            List of names of fluid particle arrays.
        solids : list(str)
            List of names of solid particle arrays.
        dim: int
            Dimensionality of the problem.
        c0 : float
            Speed of sound.
        nu : float
            Kinematic viscosity.
        rho0 : float
            Density of fluid.
        pb : float
            Background pressure value, if unset or zero, this uses an
            different formulation, else it uses the TVF with EDAC.
        gx, gy, gz : float
            Componenents of body acceleration (gravity, external forcing etc.)
        tdamp: float
            Time for which the acceleration should be damped.
        eps : float
            XSPH smoothing factor, defaults to zero.
        h : float
            Parameter h used for the particles -- used to calculate viscosity.
        edac_alpha : float
            Factor to use for viscosity.
        alpha : float
            Factor to use for artificial viscosity.
        bql : bool
            Use the Basa Quinlan Lastiwka correction.
        clamp_p : bool
            Clamp the boundary pressure to positive values.  This is only used
            for external flows.
        inlet_outlet_manager : InletOutletManager Instance
            Pass the manager if inlet outlet boundaries are present
        inviscid_solids : list
            list of inviscid solid array names
        """
        self.c0 = c0
        self.nu = nu
        self.rho0 = rho0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        self.dim = dim
        self.eps = eps
        self.fluids = fluids
        self.solids = solids
        self.pb = pb
        self.solver = None
        self.bql = bql
        self.clamp_p = clamp_p
        self.edac_alpha = edac_alpha
        self.alpha = alpha
        self.h = h
        self.inlet_outlet_manager = inlet_outlet_manager
        self.inviscid_solids = [] if inviscid_solids is None else\
            inviscid_solids
        self.attributes_changed()

    # Public protocol ###################################################
    def add_user_options(self, group):
        group.add_argument(
            "--alpha", action="store", type=float, dest="alpha",
            default=None,
            help="Alpha for the artificial viscosity."
        )
        group.add_argument(
            "--edac-alpha", action="store", type=float, dest="edac_alpha",
            default=None,
            help="Alpha for the EDAC scheme viscosity."
        )
        add_bool_argument(
            group, 'clamp-pressure', dest='clamp_p',
            help="Clamp pressure on boundaries to be non-negative.",
            default=None
        )
        add_bool_argument(
            group, 'use-bql', dest='bql',
            help="Use the Basa-Quinlan-Lastiwka correction.",
            default=None
        )
        group.add_argument(
            "--tdamp", action="store", type=float, dest="tdamp",
            default=None,
            help="Time for which the accelerations are damped."
        )

    def consume_user_options(self, options):
        vars = ['alpha', 'edac_alpha', 'clamp_p', 'bql', 'tdamp']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.pb is not None:
            self.use_tvf = abs(self.pb) > 1e-14
        if self.h is not None and self.c0 is not None:
            self.art_nu = self.edac_alpha*self.h*self.c0/8

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """Configure the solver to be generated.

        This is to be called before `get_solver` is called.

        Parameters
        ----------

        dim : int
            Number of dimensions.
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
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.integrator import PECIntegrator
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = EDACTVFStep if self.use_tvf else EDACStep
        cls = integrator_cls if integrator_cls is not None else PECIntegrator

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        iom = self.inlet_outlet_manager
        if iom is not None:
            iom_stepper = iom.get_stepper(self, cls, self.use_tvf)
            for name in iom_stepper:
                steppers[name] = iom_stepper[name]

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

        if iom is not None:
            iom.setup_iom(dim=self.dim, kernel=kernel)

    def get_equations(self):
        if self.use_tvf:
            return self._get_internal_flow_equations()
        else:
            return self._get_external_flow_equations()

    def get_solver(self):
        return self.solver

    def setup_properties(self, particles, clean=True):
        """Setup the particle arrays so they have the right set of properties
        for this scheme.

        Parameters
        ----------

        particles : list
            List of particle arrays.

        clean : bool
            If True, removes any unnecessary properties.
        """
        particle_arrays = dict([(p.name, p) for p in particles])
        TVF_FLUID_PROPS = set([
            'uhat', 'vhat', 'what', 'ap',
            'auhat', 'avhat', 'awhat', 'V',
            'p0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
            'pavg', 'nnbr'
        ])
        extra_props = TVF_FLUID_PROPS if self.use_tvf else EDAC_PROPS

        all_fluid_props = DEFAULT_PROPS.union(extra_props)
        iom = self.inlet_outlet_manager
        fluids_with_io = self.fluids
        if iom is not None:
            io_particles = iom.get_io_names(ghost=True)
            fluids_with_io = self.fluids + io_particles
        for fluid in fluids_with_io:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, all_fluid_props, clean)
            pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                                  'm', 'h', 'V'])
            if 'pavg' in pa.properties:
                pa.add_output_arrays(['pavg'])
            if iom is not None:
                iom.add_io_properties(pa, self)

        TVF_SOLID_PROPS = ['V', 'wij', 'ax', 'ay', 'az', 'uf', 'vf', 'wf',
                           'ug', 'vg', 'wg']
        if self.inviscid_solids:
            TVF_SOLID_PROPS += ['xn', 'yn', 'zn', 'uhat', 'vhat', 'what']
        extra_props = TVF_SOLID_PROPS if self.use_tvf else EDAC_SOLID_PROPS
        all_solid_props = DEFAULT_PROPS.union(extra_props)
        for solid in (self.solids+self.inviscid_solids):
            pa = particle_arrays[solid]
            self._ensure_properties(pa, all_solid_props, clean)
            pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                                  'm', 'h', 'V'])

    # Private protocol ###################################################
    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu

    def _get_internal_flow_equations(self):
        from pysph.sph.wc.transport_velocity import (
            VolumeSummation, SolidWallNoSlipBC, SummationDensity,
            MomentumEquationArtificialStress,
            MomentumEquationArtificialViscosity,
            MomentumEquationViscosity
        )
        edac_nu = self._get_edac_nu()

        iom = self.inlet_outlet_manager
        fluids_with_io = self.fluids
        all_solids = self.solids + self.inviscid_solids
        if iom is not None:
            fluids_with_io = self.fluids + iom.get_io_names()
        all = fluids_with_io + all_solids

        equations = []
        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations(self, self.use_tvf)
            for grp in io_eqns:
                equations.append(grp)

        group1 = []
        avg_p_group = []
        has_solids = len(all_solids) > 0
        for fluid in fluids_with_io:
            group1.append(SummationDensity(dest=fluid, sources=all))
            if self.bql:
                eq = ComputeAveragePressure(dest=fluid, sources=all)
                if has_solids:
                    avg_p_group.append(eq)
                else:
                    group1.append(eq)

        for solid in self.solids:
            group1.extend([
                SourceNumberDensity(dest=solid, sources=fluids_with_io),
                VolumeSummation(dest=solid, sources=all),
                SolidWallPressureBC(dest=solid, sources=fluids_with_io,
                                    gx=self.gx, gy=self.gy, gz=self.gz),
                SetWallVelocity(dest=solid, sources=fluids_with_io),
            ])
        for solid in self.inviscid_solids:
            group1.extend([
                SourceNumberDensity(dest=solid, sources=fluids_with_io),
                NoSlipVelocityExtrapolation(
                    dest=solid, sources=fluids_with_io),
                NoSlipAdvVelocityExtrapolation(
                    dest=solid, sources=fluids_with_io),
                VolumeSummation(dest=solid, sources=all),
                SolidWallPressureBC(dest=solid, sources=fluids_with_io,
                                    gx=self.gx, gy=self.gy, gz=self.gz)
                ])

        equations.append(Group(equations=group1, real=False))

        # Compute average pressure *after* the wall pressure is setup.
        if self.bql and has_solids:
            equations.append(Group(equations=avg_p_group, real=True))

        group2 = []
        for fluid in self.fluids:
            group2.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, pb=self.pb,
                    gx=self.gx, gy=self.gy, gz=self.gz, tdamp=self.tdamp
                )
            )
            if self.alpha > 0.0:
                sources = fluids_with_io + self.solids
                group2.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=sources, alpha=self.alpha,
                        c0=self.c0
                    )
                )
            if self.nu > 0.0:
                group2.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=fluids_with_io, nu=self.nu
                    )
                )
            if len(self.solids) > 0 and self.nu > 0.0:
                group2.append(
                    SolidWallNoSlipBC(
                        dest=fluid, sources=self.solids, nu=self.nu
                    )
                )
            group2.extend([
                MomentumEquationArtificialStress(
                    dest=fluid, sources=fluids_with_io
                ),
                EDACEquation(
                    dest=fluid, sources=all, nu=edac_nu, cs=self.c0,
                    rho0=self.rho0
                ),
            ])
        equations.append(Group(equations=group2))

        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations_post_compute_acceleration()
            for grp in io_eqns:
                equations.append(grp)

        return equations

    def _get_external_flow_equations(self):

        from pysph.sph.basic_equations import XSPHCorrection
        from pysph.sph.wc.transport_velocity import (
            VolumeSummation, SolidWallNoSlipBC, SummationDensity,
            MomentumEquationArtificialViscosity,
            MomentumEquationViscosity
        )

        iom = self.inlet_outlet_manager
        fluids_with_io = self.fluids
        all_solids = self.solids + self.inviscid_solids
        if iom is not None:
            fluids_with_io = self.fluids + iom.get_io_names()
        all = fluids_with_io + all_solids

        edac_nu = self._get_edac_nu()
        equations = []
        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations(self, self.use_tvf)
            for grp in io_eqns:
                equations.append(grp)

        group1 = []
        for fluid in fluids_with_io:
            group1.append(SummationDensity(dest=fluid, sources=all))
        for solid in self.solids:
            group1.extend([
                SourceNumberDensity(dest=solid, sources=fluids_with_io),
                VolumeSummation(dest=solid, sources=all),
                SolidWallPressureBC(dest=solid, sources=fluids_with_io,
                                    gx=self.gx, gy=self.gy, gz=self.gz),
                SetWallVelocity(dest=solid, sources=fluids_with_io),
            ])
            if self.clamp_p:
                group1.append(
                    ClampWallPressure(dest=solid, sources=None)
                )

        for solid in self.inviscid_solids:
            group1.extend([
                SourceNumberDensity(dest=solid, sources=fluids_with_io),
                NoSlipVelocityExtrapolation(
                    dest=solid, sources=fluids_with_io),
                VolumeSummation(dest=solid, sources=all),
                SolidWallPressureBC(dest=solid, sources=fluids_with_io,
                                    gx=self.gx, gy=self.gy, gz=self.gz)
                ])

        equations.append(Group(equations=group1, real=False))

        group2 = []
        for fluid in self.fluids:
            group2.append(
                MomentumEquation(
                    dest=fluid, sources=all, gx=self.gx, gy=self.gy,
                    gz=self.gz, c0=self.c0, tdamp=self.tdamp
                )
            )
            if self.alpha > 0.0:
                sources = fluids_with_io + self.solids
                group2.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=sources, alpha=self.alpha,
                        c0=self.c0
                    )
                )
            if self.nu > 0.0:
                group2.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=fluids_with_io, nu=self.nu
                    )
                )
            if len(self.solids) > 0 and self.nu > 0.0:
                group2.append(
                    SolidWallNoSlipBC(
                        dest=fluid, sources=self.solids, nu=self.nu
                    )
                )
            group2.extend([
                EDACEquation(
                    dest=fluid, sources=all, nu=edac_nu, cs=self.c0,
                    rho0=self.rho0
                ),
                XSPHCorrection(dest=fluid, sources=[fluid],
                               eps=self.eps)
            ])
        equations.append(Group(equations=group2))

        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations_post_compute_acceleration()
            for grp in io_eqns:
                equations.append(grp)

        return equations
