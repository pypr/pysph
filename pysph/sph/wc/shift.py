"""
Shift particle positions
########################

Equations to maintain uniform particle distribution.

There are two ways proposed:
1. 'SimpleShift' is the earlier one, see [XuStaLau2009].
2. 'FickianShift' is based on Fick's diffusion law, see [LiXuStaRo2012].

TODO: Implement for free surface.

References
----------

    .. [XuStaLau2009] Rui Xu, Peter Stansby, Dominique Laurence "Accuracy
        and stability in incompressible SPH (ISPH) based on the projection
        method and a new approach", Journal of Computational Physics 228
        (2009), pp. 6703--6725.

    .. [LiXuStaRo2012] S.J Lind, R. Xu, P.K. Stansby, B.D. Rogers
        "Incompressible smoothed particle hydrodynamics for free-surface flows:
        A generalised diffusion-based algorithm for stability and validations
        for impulsive flows and propagating waves", Journal of Computational
        Physics 231 (2009), pp. 1499--1523.

    .. [SkLiStaRo2013] Alex Skillen, S. Lind, P.K. Stansby, B.D. Rogers
        "Incompressible smoothed particle hydrodynamics (SPH) with reduced
        temporal noise and generalised Fickian smoothing applied to
        body-water slam and efficient wave-body interaction", Computer Methods
        in Applied Mechanics and Engineering 265 (2013), pp. 163--173.
"""

from math import sqrt
from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.base.reduce_array import parallel_reduce_array, serial_reduce_array
from pysph.solver.tools import Tool


class SimpleShift(Equation):
    r"""**Simple shift**
    See the paper [XuStaLau2009], equation(35)
    """
    def __init__(self, dest, sources, const=0.04):
        self.beta = const
        super(SimpleShift, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        from numpy import sqrt
        vmag = sqrt(dst.u**2 + dst.v**2 + dst.w**2)
        dst.vmax[0] = serial_reduce_array(vmag, 'max')
        dst.vmax[:] = parallel_reduce_array(dst.vmax, 'max')

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_vmax, d_dpos, dt,
                 N_NBRS, NBRS):
        i, s_idx = declare('int', 2)
        ri = 0.0
        dxi = 0.0
        dyi = 0.0
        dzi = 0.0
        eps = 1.0e-08
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij*xij + yij*yij + zij*zij)
            r3ij = rij * rij * rij
            dxi += xij / (r3ij + eps)
            dyi += yij / (r3ij + eps)
            dzi += zij / (r3ij + eps)
            ri += rij

        ri = ri/N_NBRS
        fac = self.beta * ri*ri * d_vmax[0] * dt
        d_dpos[d_idx*3] = fac*dxi
        d_dpos[d_idx*3 + 1] = fac*dyi
        d_dpos[d_idx*3 + 2] = fac*dzi

        d_x[d_idx] += d_dpos[d_idx*3]
        d_y[d_idx] += d_dpos[d_idx*3 + 1]
        d_z[d_idx] += d_dpos[d_idx*3 + 2]


class FickianShift(Equation):
    r"""**Fickian-shift**
    See the paper [LiXuStaRo2012], equation(21-24),
    for the constant see [SkLiStaRo2013], equation(13).
    """
    def __init__(self, dest, sources, fickian_const=10, tensile_const=0.2,
                 tensile_pow=4, hdx=1.0, tensile_correction=False):
        self.fickian_const = fickian_const
        self.tensile_const = tensile_const
        self.tensile_pow = tensile_pow
        self.hdx = hdx
        self.tensile_correction = tensile_correction
        super(FickianShift, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_u, d_v, d_w, d_h,
                 s_h, s_m, s_rho, dt, d_dpos,
                 N_NBRS, NBRS, SPH_KERNEL):
        i, s_idx = declare('int', 2)
        xij, dwij, grad_c = declare('matrix(3)', 3)
        grad_c[0] = 0.0
        grad_c[1] = 0.0
        grad_c[2] = 0.0
        ui = d_u[d_idx]
        vi = d_v[d_idx]
        wi = d_w[d_idx]
        vmag = sqrt(ui*ui + vi*vi + wi*wi)

        hi = d_h[d_idx]
        dx = declare('matrix(3)')
        dx[0] = hi/self.hdx
        dx[1] = 0.0
        dx[2] = 0.0
        fij = 0.0
        wdx = SPH_KERNEL.kernel(dx, dx[0], d_h[d_idx])

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])
            hij = (hi + s_h[s_idx]) * 0.5
            SPH_KERNEL.gradient(xij, rij, hij, dwij)
            Vj = s_m[s_idx] / s_rho[s_idx]

            if self.tensile_correction:
                R = self.tensile_const
                n = self.tensile_pow
                wij = SPH_KERNEL.kernel(xij, rij, hij)
                fij = R * (wij/wdx)**n

            grad_c[0] += Vj * (1 + fij) * dwij[0]
            grad_c[1] += Vj * (1 + fij) * dwij[1]
            grad_c[2] += Vj * (1 + fij) * dwij[2]

        fac = -self.fickian_const * hi * vmag * dt
        d_dpos[d_idx*3] = fac*grad_c[0]
        d_dpos[d_idx*3 + 1] = fac*grad_c[1]
        d_dpos[d_idx*3 + 2] = fac*grad_c[2]

        d_x[d_idx] += d_dpos[d_idx*3]
        d_y[d_idx] += d_dpos[d_idx*3 + 1]
        d_z[d_idx] += d_dpos[d_idx*3 + 2]


class CorrectVelocities(Equation):
    r"""**Correct velocities**
    Correct the velocities after shifting to a new position by using taylor
    series approximation, see equation (34) of [XuStaLau2009].

    .. math::
        \phi_{i}^' = \phi_i + (\nabla \phi)_i \cdot \delta \mathbf{r}_{ii^'}

    where, \phi_{i} is the hydrodynamic variable at old position, \phi_{i}^'
    is at new position, delta \mathbf{r}_{ii^'} is the vector between new and
    old position.
    """
    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx + i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_gradv, DWIJ, VIJ):
        alp, bet = declare('int', 2)

        Vj = s_m[s_idx] / s_rho[s_idx]

        for alp in range(3):
            for bet in range(3):
                d_gradv[d_idx*9 + 3*bet + alp] += -Vj * VIJ[alp] * DWIJ[bet]

    def post_loop(self, d_idx, d_u, d_v, d_w, d_gradv, d_dpos):
        res = declare('matrix(3)')
        i, j = declare('int', 2)
        for i in range(3):
            tmp = 0.0
            for j in range(3):
                tmp += d_gradv[d_idx*9 + 3*i + j] * d_dpos[d_idx*3 + j]
            res[i] = tmp

        d_u[d_idx] += res[0]
        d_v[d_idx] += res[1]
        d_w[d_idx] += res[2]


class ShiftPositions(Tool):
    def __init__(self, app, array_name, freq=1, shift_kind='simple',
                 correct_velocity=False, parameter=None):
        """
        Parameters
        ----------

        app : pysph.solver.application.Application.
            The application instance.
        arr_name : array
            Name of the particle array whose position needs to be
            shifted.
        freq : int
            Frequency to apply particle position shift.
        shift_kind: str
            Kind to shift to apply available are "simple" and "fickian".
        correct_velocity: bool
            Correct velocities after shift in particle position.
        parameter: float
            Correct velocities after shift in particle position.
        """
        from pysph.solver.utils import get_array_by_name
        self.particles = app.particles
        self.dt = app.solver.dt
        self.dim = app.solver.dim
        self.kernel = app.solver.kernel
        self.array = get_array_by_name(self.particles, array_name)
        self.freq = freq
        self.kind = shift_kind
        self.correct_velocity = correct_velocity
        self.parameter = parameter
        self.count = 1
        self._sph_eval = None
        options = ['simple', 'fickian']
        assert self.kind in options, 'shift_kind should be one of %s' % options

    def _get_sph_eval(self, kind):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        if self._sph_eval is None:
            arr = self.array
            eqns = []
            name = arr.name
            if 'vmax' not in arr.constants.keys():
                arr.add_constant('vmax', [0.0])
            if 'dpos' not in arr.properties.keys():
                arr.add_property('dpos', stride=3)
            if kind == 'simple':
                const = 0.04 if not self.parameter else self.parameter
                eqns.append(Group(
                    equations=[SimpleShift(name, [name], const=const)],
                    update_nnps=True)
                )
            elif kind == 'fickian':
                const = 4 if not self.parameter else self.parameter
                eqns.append(Group(
                    equations=[FickianShift(name, [name],
                                            fickian_const=const)],
                    update_nnps=True)
                )
            if self.correct_velocity:
                if 'gradv' not in arr.properties.keys():
                    arr.add_property('gradv', stride=9)
                eqns.append(Group(equations=[
                    CorrectVelocities(name, [name])]))

            sph_eval = SPHEvaluator(
                arrays=[arr], equations=eqns, dim=self.dim,
                kernel=self.kernel)
            return sph_eval
        else:
            return self._sph_eval

    def post_step(self, solver):
        if self.freq == 0:
            pass
        elif self.count % self.freq == 0:
            self._sph_eval = self._get_sph_eval(self.kind)
            self._sph_eval.update()
            self._sph_eval.evaluate(dt=self.dt)
        self.count += 1
