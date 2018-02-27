from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.reduce_array import serial_reduce_array


def get_particle_array_pcisph(dt, dim, delta=None, constants=None, **props):
    from pysph.sph.equation import Group
    from pysph.base.kernels import CubicSpline
    pcisph_props = [
        'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'ao_x', 'ao_y', 'ao_z', 'ap_x',
        'ap_y', 'ap_z', 'rho_err', 'drho', 'sum_dwij_x', 'sum_dwij_y',
        'sum_dwij_z', 'sum_2dwij', 'local_delta', 'tmp_comp'
    ]

    # compression factor
    consts = {'delta': [0.0], 'rho_base': [0.0]}
    if constants:
        consts.update(constants)

    print(dim)
    pa = get_particle_array(constants=consts, additional_props=pcisph_props,
                            **props)
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid'
    ])

    equations = [
        Group(equations=[
            ComputeDelta(dest=pa.name, sources=[pa.name], dt=dt),
        ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=dim,
                            kernel=CubicSpline(dim=dim))
    sph_eval.evaluate()

    if delta:
        pa.delta[0] = delta

    print("Delta is ")
    print(pa.delta[0])
    return pa


class ComputeDelta(Equation):
    def __init__(self, dest, sources, dt=1e-3):
        self.dt = dt
        super(ComputeDelta, self).__init__(dest, sources)

    def loop(self, d_idx, d_sum_dwij_x, d_sum_dwij_y, d_sum_dwij_z,
             d_sum_2dwij, DWIJ):
        d_sum_2dwij[d_idx] += (
            DWIJ[0] * DWIJ[0] + DWIJ[1] * DWIJ[1] + DWIJ[2] * DWIJ[2])
        d_sum_dwij_x[d_idx] += DWIJ[0]
        d_sum_dwij_y[d_idx] += DWIJ[1]
        d_sum_dwij_z[d_idx] += DWIJ[2]

    def post_loop(self, d_idx, d_local_delta, d_m, d_rho_base, d_sum_dwij_x,
                  d_sum_dwij_y, d_sum_dwij_z, d_sum_2dwij):
        beta = 2. * d_m[d_idx]**2 * self.dt**2. / d_rho_base[0]**2
        d_local_delta[d_idx] = 1. / (beta * (
            d_sum_2dwij[d_idx] +
            (d_sum_dwij_x[d_idx] * d_sum_dwij_x[d_idx] + d_sum_dwij_y[d_idx] *
             d_sum_dwij_y[d_idx] + d_sum_dwij_z[d_idx] * d_sum_dwij_z[d_idx])))

    def reduce(self, dst):
        dst.delta[0] = serial_reduce_array(dst.local_delta, 'max')


def get_particle_array_static_boundary(constants=None, **props):
    extra_props = ['x', 'y', 'z', 'V']

    # compression factor
    consts = {}
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid'
    ])

    return pa


class SaveCurrentProps(Equation):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_x, d_y,
                   d_z, d_u, d_v, d_w):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_u0[d_idx] = d_u[d_idx]
        d_w0[d_idx] = d_w[d_idx]


class ClearAccelerationsAddGravity(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(ClearAccelerationsAddGravity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ao_x, d_ao_y, d_ao_z, d_ap_x, d_ap_y,
                   d_ap_z):
        d_ao_x[d_idx] = self.gx
        d_ao_y[d_idx] = self.gy
        d_ao_z[d_idx] = self.gz
        d_ap_x[d_idx] = 0
        d_ap_y[d_idx] = 0
        d_ap_z[d_idx] = 0


class ComputeDensityFluid(Equation):
    def __init__(self, dest, sources):
        super(ComputeDensityFluid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0

    def loop(self, d_idx, s_idx, d_rho, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx] * WIJ


class ComputeDensitySolid(Equation):
    def loop(self, d_idx, s_idx, d_rho_base, d_rho, s_V, WIJ):
        d_rho[d_idx] += s_V[s_idx] * d_rho_base[0] * WIJ


class ComputeNonPressureForces(Equation):
    pass


class SetUpPressureSolver(Equation):
    def initialize(self, d_idx, d_p, d_ap_x, d_ap_y, d_ap_z):
        d_p[d_idx] = 0
        d_ap_x[d_idx] = 0
        d_ap_y[d_idx] = 0
        d_ap_z[d_idx] = 0


class Advect(Equation):
    def initialize(self, d_idx, d_ao_x, d_ao_y, d_ao_z, d_ap_x, d_ap_y, d_ap_z,
                   d_u, d_v, d_w, d_u0, d_v0, d_w0, d_x, d_y, d_z, d_x0, d_y0,
                   d_z0, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * (d_ap_x[d_idx] + d_ao_x[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dt * (d_ap_y[d_idx] + d_ao_y[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dt * (d_ap_z[d_idx] + d_ao_z[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]


class DensityDifference(Equation):
    def __init__(self, dest, sources, eta=10, debug=False):
        super(DensityDifference, self).__init__(dest, sources)
        self.eta = eta
        self.debug = debug
        self.compression = 0.

    def post_loop(self, d_idx, d_rho, d_drho, d_rho_base):
        d_drho[d_idx] = max(d_rho[d_idx] - d_rho_base[0], 0)

    def reduce(self, dst):
        # first sum the density difference of all particles
        dst.tmp_comp[0] = serial_reduce_array(dst.drho, 'sum') / len(dst.drho)
        self.compression = dst.tmp_comp[0]

    def converged(self):
        debug = self.debug
        compression = self.compression

        if compression > self.eta:
            if debug:
                print("Not Converged", compression)
            return -1.0
        else:
            if debug:
                print("Converged", compression)
            return -1.0


class CorrectPressure(Equation):
    def post_loop(self, d_idx, d_drho, d_p, d_delta):
        d_p[d_idx] += d_delta[0] * d_drho[d_idx]


class MomentumEquation(Equation):
    def initialize(self, d_idx, d_ap_x, d_ap_y, d_ap_z, DWIJ):
        d_ap_x[d_idx] = 0
        d_ap_y[d_idx] = 0
        d_ap_z[d_idx] = 0

    def loop(self, d_idx, s_idx, d_rho_base, d_p, d_ap_x, d_ap_y, d_ap_z, s_p,
             s_m, DWIJ):
        dpi = d_p[d_idx] / (d_rho_base[0]**2)
        dpj = s_p[s_idx] / (d_rho_base[0]**2)
        d_ap_x[d_idx] += -s_m[s_idx] * (dpi + dpj) * DWIJ[0]
        d_ap_y[d_idx] += -s_m[s_idx] * (dpi + dpj) * DWIJ[1]
        d_ap_z[d_idx] += -s_m[s_idx] * (dpi + dpj) * DWIJ[2]


class MomentumEquationStaticBoundary(Equation):
    def loop(self, d_idx, s_idx, d_rho_base, d_p, d_ap_x, d_ap_y, d_ap_z, s_p,
             s_V, DWIJ):
        dpi = d_p[d_idx] / (d_rho_base[0]**2)
        d_ap_x[d_idx] += -s_V[s_idx] * d_rho_base[0] * dpi * DWIJ[0]
        d_ap_y[d_idx] += -s_V[s_idx] * d_rho_base[0] * dpi * DWIJ[1]
        d_ap_z[d_idx] += -s_V[s_idx] * d_rho_base[0] * dpi * DWIJ[2]


class PCISPHStep(IntegratorStep):
    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_ap_x, d_ap_y, d_ap_z, d_ao_x,
               d_ao_y, d_ao_z, d_u, d_v, d_w, dt):
        d_u[d_idx] = d_u[d_idx] + dt * (d_ap_x[d_idx] + d_ao_x[d_idx])
        d_v[d_idx] = d_v[d_idx] + dt * (d_ap_y[d_idx] + d_ao_y[d_idx])
        d_w[d_idx] = d_w[d_idx] + dt * (d_ap_z[d_idx] + d_ao_z[d_idx])

        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]
