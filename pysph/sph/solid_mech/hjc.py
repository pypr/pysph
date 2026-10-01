"""Holmquist--Johnson--Cook concrete for the solid mechanics equations.

Use :class:`HJCStep` with EPECIntegrator and HookesDeviatoricStressRate.
The pressure is updated by the stepper; do not add a separate EOS equation.
See ``docs/source/examples/hjc.rst`` for conventions and references.
"""
from math import log, pow, sqrt

import numpy as np
from compyle.api import annotate, declare
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics


def hjc_parameters(rho0, G, A, B, C, N, fc, T, edot0, efmin, sfmax,
                   pc, muc, pl, mul, D1, D2, K1, K2, K3):
    """Pack a validated material in consistent units.

    ``mul`` is permanent compaction at locking, not total compression.
    ``pc/muc`` is the initial bulk modulus. The dense EOS and crushing
    line meet at ``pl``. See the example for an illustrative parameter set.
    """
    p = np.array([rho0, G, A, B, C, N, fc, T, edot0, efmin, sfmax,
                  pc, muc, pl, mul, D1, D2, K1, K2, K3], dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('HJC parameters must be finite')
    positive = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]
    nonnegative = [2, 3, 4, 16, 18, 19]
    if (np.any(p[positive] <= 0) or np.any(p[nonnegative] < 0)
            or pl <= pc):
        raise ValueError('Invalid HJC parameter range')
    lo, hi = 0.0, pl / K1
    for _ in range(64):
        eta = (lo + hi) / 2
        if eta * (K1 + eta * (K2 + eta * K3)) < pl:
            lo = eta
        else:
            hi = eta
    lock = mul + (1 + mul) * (lo + hi) / 2
    if lock <= muc:
        raise ValueError('Locking must follow crushing')
    slope = (pl - pc) / (lock - muc)
    if slope >= min(pc / muc, pl / (lock - mul)):
        raise ValueError('Crushing must be softer than elastic unloading')
    result = np.append(p, [lock, slope])
    if not np.isfinite(result).all() or not np.isfinite(pc / muc):
        raise ValueError('HJC derived moduli must be finite')
    return result


@annotate(q='double', mu='double', rate='double', damage='double',
          ep='double', plastic_volume='double', peak='double',
          p='doublep', result='doublep')
def hjc_return(q, mu, rate, damage, ep, plastic_volume, peak, p, result):
    """Return a trial equivalent stress and update irreversible history.

    ``result`` stores scale, damage, plastic strain, plastic compression,
    peak compression, pressure, longitudinal sound speed, equivalent stress.
    The supplied history is always the start-of-step state.
    """
    rho0, G, A, B, C, N = p[0], p[1], p[2], p[3], p[4], p[5]
    fc, T, edot0, efmin, sfmax = p[6], p[7], p[8], p[9], p[10]
    pc, muc, pl, mul = p[11], p[12], p[13], p[14]
    D1, D2, K1, K2, K3 = p[15], p[16], p[17], p[18], p[19]
    lock, slope = p[20], p[21]
    peak = max(peak, mu)
    volume = mul * min(1.0, max(0.0, (peak - muc) / (lock - muc)))
    bulk = pc / muc
    if peak >= lock:
        eta = (mu - mul) / (1.0 + mul)
        pressure = K1 * eta
        if eta > 0.0:
            pressure += eta * eta * (K2 + eta * K3)
    elif peak > muc:
        unload = (pc + slope * (peak - muc)) / (peak - volume)
        pressure = unload * (mu - volume)
    else:
        pressure = bulk * mu
    pressure = max(pressure, -T * (1.0 - damage))
    strength = A * (1.0 - damage) + B * pow(max(pressure, 0.0) / fc, N)
    if pressure < 0.0:
        strength = A * max(0.0, 1.0 - damage + pressure / T)
    factor = 1.0 + C * log(max(1.0, rate / edot0))
    strength = fc * min(sfmax, strength * factor)
    scale = 1.0
    if q > strength:
        scale = strength / q
    dep = (1.0 - scale) * q / (3.0 * G)
    failure = max(efmin, D1 * pow(max(0.0, (pressure + T) / fc), D2))
    damage = min(1.0, damage + (dep + max(0.0, volume - plastic_volume))
                 / failure)
    # A conservative wave speed also covers unloading from past compression.
    eta = max(0.0, (peak - mul) / (1.0 + mul))
    dense = (K1 + 2.0 * K2 * eta + 3.0 * K3 * eta * eta) / (1.0 + mul)
    modulus = (1.0 + max(0.0, peak)) * max(bulk, dense) + 4.0 * G / 3.0
    result[0] = scale
    result[1] = damage
    result[2] = ep + dep
    result[3] = volume
    result[4] = peak
    result[5] = max(pressure, -T * (1.0 - damage))
    result[6] = sqrt(modulus / (rho0 * (1.0 + mu)))
    result[7] = q * scale


def get_particle_array_hjc(parameters, **props):
    """Create particles with zero initial damage and saved stage histories."""
    p = hjc_parameters(*np.asarray(parameters)[:20])
    bulk, G = p[11] / p[12], p[1]
    constants = dict(props.pop('constants', {}))
    if set(constants) & {'rho_ref', 'E', 'nu', 'G', 'hjc', 'c0_ref'}:
        raise ValueError('Specify material constants through hjc_parameters')
    constants.update(rho_ref=p[0], E=9 * bulk * G / (3 * bulk + G),
                     nu=(3 * bulk - 2 * G) / (2 * (3 * bulk + G)), hjc=p)
    props.setdefault('rho', p[0])
    pa = get_particle_array_elastic_dynamics(constants=constants, **props)
    for name in ('damage', 'plastic_strain', 'plastic_volume',
                 'max_compression'):
        pa.add_property(name)
        pa.add_property(name + '0')
    pa.add_property('q')
    pa.add_property('dt_cfl')
    pa.add_output_arrays(['s00', 's01', 's02', 's11', 's12', 's22', 'q',
                          'damage', 'plastic_strain', 'plastic_volume',
                          'max_compression', 'cs'])
    pa.cs[:] = sqrt((max(bulk, p[17] / (1 + p[14])) + 4 * G / 3) / p[0])
    pa.dt_cfl[:] = pa.cs
    return pa


class HJCStep(IntegratorStep):
    """EPEC solid step with a radial return from the saved step history.

    Both stages restart from the same history. Equation evaluations have
    no irreversible side effects. The Jaumann trial stress is supplied by
    HookesDeviatoricStressRate; tensor components include the out-of-plane
    normal stress in a 2D plane-strain calculation.
    """
    def _get_helpers_(self):
        return [hjc_return]

    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
                   d_u, d_v, d_w, d_u0, d_v0, d_w0, d_rho, d_rho0,
                   d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_s000, d_s010, d_s020, d_s110, d_s120, d_s220,
                   d_damage, d_damage0, d_plastic_strain, d_plastic_strain0,
                   d_plastic_volume, d_plastic_volume0,
                   d_max_compression, d_max_compression0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]
        d_s000[d_idx] = d_s00[d_idx]
        d_s010[d_idx] = d_s01[d_idx]
        d_s020[d_idx] = d_s02[d_idx]
        d_s110[d_idx] = d_s11[d_idx]
        d_s120[d_idx] = d_s12[d_idx]
        d_s220[d_idx] = d_s22[d_idx]
        d_damage0[d_idx] = d_damage[d_idx]
        d_plastic_strain0[d_idx] = d_plastic_strain[d_idx]
        d_plastic_volume0[d_idx] = d_plastic_volume[d_idx]
        d_max_compression0[d_idx] = d_max_compression[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_u, d_v, d_w, d_u0, d_v0, d_w0, d_au, d_av, d_aw,
               d_rho, d_rho0, d_arho, d_s00, d_s01, d_s02,
               d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
               d_s110, d_s120, d_s220, d_as00, d_as01, d_as02,
               d_as11, d_as12, d_as22, d_v00, d_v01, d_v02,
               d_v10, d_v11, d_v12, d_v20, d_v21, d_v22,
               d_damage, d_damage0, d_plastic_strain, d_plastic_strain0,
               d_plastic_volume, d_plastic_volume0,
               d_max_compression, d_max_compression0,
               d_p, d_cs, d_q, d_dt_cfl, d_hjc, dt):
        self.stage2(d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
                    d_u, d_v, d_w, d_u0, d_v0, d_w0, d_au, d_av, d_aw,
                    d_rho, d_rho0, d_arho, d_s00, d_s01, d_s02,
                    d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
                    d_s110, d_s120, d_s220, d_as00, d_as01, d_as02,
                    d_as11, d_as12, d_as22, d_v00, d_v01, d_v02,
                    d_v10, d_v11, d_v12, d_v20, d_v21, d_v22,
                    d_damage, d_damage0, d_plastic_strain, d_plastic_strain0,
                    d_plastic_volume, d_plastic_volume0,
                    d_max_compression, d_max_compression0,
                    d_p, d_cs, d_q, d_dt_cfl, d_hjc, 0.5 * dt)

    def stage2(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
               d_u, d_v, d_w, d_u0, d_v0, d_w0, d_au, d_av, d_aw,
               d_rho, d_rho0, d_arho, d_s00, d_s01, d_s02,
               d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
               d_s110, d_s120, d_s220, d_as00, d_as01, d_as02,
               d_as11, d_as12, d_as22, d_v00, d_v01, d_v02,
               d_v10, d_v11, d_v12, d_v20, d_v21, d_v22,
               d_damage, d_damage0, d_plastic_strain, d_plastic_strain0,
               d_plastic_volume, d_plastic_volume0,
               d_max_compression, d_max_compression0,
               d_p, d_cs, d_q, d_dt_cfl, d_hjc, dt):
        result = declare('matrix(8)')
        # Use the velocity evaluated at this stage, before its update.
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        s00 = d_s000[d_idx] + dt * d_as00[d_idx]
        s01 = d_s010[d_idx] + dt * d_as01[d_idx]
        s02 = d_s020[d_idx] + dt * d_as02[d_idx]
        s11 = d_s110[d_idx] + dt * d_as11[d_idx]
        s12 = d_s120[d_idx] + dt * d_as12[d_idx]
        s22 = d_s220[d_idx] + dt * d_as22[d_idx]
        mean = (s00 + s11 + s22) / 3.0
        s00 -= mean
        s11 -= mean
        s22 -= mean
        q = sqrt(1.5 * (s00*s00 + s11*s11 + s22*s22
                        + 2.0 * (s01*s01 + s02*s02 + s12*s12)))
        trace = (d_v00[d_idx] + d_v11[d_idx] + d_v22[d_idx]) / 3.0
        e00 = d_v00[d_idx] - trace
        e11 = d_v11[d_idx] - trace
        e22 = d_v22[d_idx] - trace
        e01 = 0.5 * (d_v01[d_idx] + d_v10[d_idx])
        e02 = 0.5 * (d_v02[d_idx] + d_v20[d_idx])
        e12 = 0.5 * (d_v12[d_idx] + d_v21[d_idx])
        norm2 = e00*e00 + e11*e11 + e22*e22
        norm2 += 2.0 * (e01*e01 + e02*e02 + e12*e12)
        rate = sqrt(2.0 / 3.0 * norm2)
        hjc_return(q, d_rho[d_idx] / d_hjc[0] - 1.0, rate,
                   d_damage0[d_idx], d_plastic_strain0[d_idx],
                   d_plastic_volume0[d_idx], d_max_compression0[d_idx],
                   d_hjc, result)
        d_s00[d_idx] = result[0] * s00
        d_s01[d_idx] = result[0] * s01
        d_s02[d_idx] = result[0] * s02
        d_s11[d_idx] = result[0] * s11
        d_s12[d_idx] = result[0] * s12
        d_s22[d_idx] = result[0] * s22
        d_damage[d_idx] = result[1]
        d_plastic_strain[d_idx] = result[2]
        d_plastic_volume[d_idx] = result[3]
        d_max_compression[d_idx] = result[4]
        d_p[d_idx] = result[5]
        d_cs[d_idx] = result[6]
        d_q[d_idx] = result[7]
        speed = sqrt(d_u[d_idx]**2 + d_v[d_idx]**2 + d_w[d_idx]**2)
        d_dt_cfl[d_idx] = result[6] + speed
