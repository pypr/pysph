import numpy as np
import pytest

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

pytest.importorskip('warp')

from cyarray.carray import UIntArray

from pysph.base.kernels import CubicSpline, Gaussian
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
import pysph.base.warp_sph as warp_sph
from pysph.base.warp_sph import (
    compute_artificial_viscosity, compute_continuity, compute_isothermal_eos,
    compute_pressure_gradient, compute_summation_density, compute_tait_eos,
    compute_wcsph_accel_continuity, compute_wcsph_adaptive_timestep,
    compute_xsph_correction, euler_step, leapfrog_drift, leapfrog_kick,
    save_wcsph_state, wc_sph_euler_step, wc_sph_leapfrog_step, wcsph_pec_stage,
    wrap_periodic
)


def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return nbrs.get_npy_array()[:nbrs.length]


def _cpu_kernel(dim, kernel='cubic'):
    if kernel == 'gaussian':
        return Gaussian(dim=dim)
    return CubicSpline(dim=dim)


def _cpu_summation_density(particles, src_index, dst_index, dim,
                           radius_scale=2.0, kernel='cubic'):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel_obj = _cpu_kernel(dim, kernel)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros(dst.get_number_of_particles())

    for d_idx in range(dst.get_number_of_particles()):
        total = 0.0
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            total += src.m[s_idx] * kernel_obj.kernel(
                xij=xij, rij=rij, h=hij
            )
        result[d_idx] = total
    return result


def _cpu_continuity(particles, src_index, dst_index, dim, radius_scale=2.0,
                    kernel='cubic'):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel_obj = _cpu_kernel(dim, kernel)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros(dst.get_number_of_particles())

    for d_idx in range(dst.get_number_of_particles()):
        total = 0.0
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            vij = [
                dst.u[d_idx] - src.u[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
                vij[1] = dst.v[d_idx] - src.v[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
                vij[2] = dst.w[d_idx] - src.w[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            dwij = [0.0, 0.0, 0.0]
            kernel_obj.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            total += src.m[s_idx] * (
                vij[0]*dwij[0] + vij[1]*dwij[1] + vij[2]*dwij[2]
            )
        result[d_idx] = total
    return result


def _cpu_pressure_gradient(particles, src_index, dst_index, dim,
                           radius_scale=2.0, kernel='cubic'):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel_obj = _cpu_kernel(dim, kernel)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros((dst.get_number_of_particles(), 3))

    for d_idx in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        rhoi21 = 1.0/(dst.rho[d_idx]*dst.rho[d_idx])
        tmpi = dst.p[d_idx]*rhoi21
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            dwij = [0.0, 0.0, 0.0]
            kernel_obj.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            rhoj21 = 1.0/(src.rho[s_idx]*src.rho[s_idx])
            tmp = tmpi + src.p[s_idx]*rhoj21
            acc += -src.m[s_idx] * tmp * np.asarray(dwij)
        result[d_idx, :] = acc
    return result


def _cpu_tait_eos(rho, rho0, c0, gamma=7.0, p0=0.0):
    ratio = rho / rho0
    p = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1.0)
    cs = c0 * ratio**(0.5 * (gamma - 1.0))
    return p, cs


def _cpu_artificial_viscosity(particles, src_index, dst_index, dim,
                              alpha, beta, c0, radius_scale=2.0,
                              kernel='cubic'):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel_obj = _cpu_kernel(dim, kernel)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros((dst.get_number_of_particles(), 3))
    src_has_cs = 'cs' in src.properties
    dst_has_cs = 'cs' in dst.properties

    for d_idx in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            vij = [
                dst.u[d_idx] - src.u[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
                vij[1] = dst.v[d_idx] - src.v[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
                vij[2] = dst.w[d_idx] - src.w[s_idx]
            vdotx = vij[0]*xij[0] + vij[1]*xij[1] + vij[2]*xij[2]
            if vdotx < 0.0:
                rij2 = xij[0]**2 + xij[1]**2 + xij[2]**2
                rij = np.sqrt(rij2)
                hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
                mu = hij * vdotx / (rij2 + 0.01*hij*hij)
                rhoij1 = 2.0 / (dst.rho[d_idx] + src.rho[s_idx])
                csi = dst.cs[d_idx] if dst_has_cs else c0
                csj = src.cs[s_idx] if src_has_cs else c0
                cij = 0.5 * (csi + csj)
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                dwij = [0.0, 0.0, 0.0]
                kernel_obj.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
                acc += -src.m[s_idx] * piij * np.asarray(dwij)
        result[d_idx, :] = acc
    return result


def _cpu_xsph_correction(particles, src_index, dst_index, dim, eps=0.5,
                         radius_scale=2.0, kernel='cubic'):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel_obj = _cpu_kernel(dim, kernel)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros((dst.get_number_of_particles(), 3))

    for d_idx in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            vij = [
                dst.u[d_idx] - src.u[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
                vij[1] = dst.v[d_idx] - src.v[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
                vij[2] = dst.w[d_idx] - src.w[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            wij = kernel_obj.kernel(xij=xij, rij=rij, h=hij)
            rhoij1 = 2.0 / (dst.rho[d_idx] + src.rho[s_idx])
            tmp = -eps * src.m[s_idx] * wij * rhoij1
            acc += tmp * np.asarray(vij)
        result[d_idx, :] = acc
    return result


def _cpu_wcsph_dt(particles, pa_index, dim, c0, cfl, dt_min, dt_max,
                  radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    pa = particles[pa_index]
    max_cfl = 0.0
    max_force = 0.0
    hmin = np.min(pa.h)
    for i in range(pa.get_number_of_particles()):
        for j in _neighbors(nnps, pa_index, pa_index, i):
            xij = [pa.x[i] - pa.x[j], 0.0, 0.0]
            vij = [pa.u[i] - pa.u[j], 0.0, 0.0]
            if dim > 1:
                xij[1] = pa.y[i] - pa.y[j]
                vij[1] = pa.v[i] - pa.v[j]
            if dim > 2:
                xij[2] = pa.z[i] - pa.z[j]
                vij[2] = pa.w[i] - pa.w[j]
            rij2 = xij[0]**2 + xij[1]**2 + xij[2]**2
            if rij2 > 1.0e-12:
                hij = 0.5 * (pa.h[i] + pa.h[j])
                vdotx = vij[0]*xij[0] + vij[1]*xij[1] + vij[2]*xij[2]
                max_cfl = max(max_cfl, abs(hij * vdotx / rij2) + c0)
        max_force = max(
            max_force,
            pa.au[i]*pa.au[i] + pa.av[i]*pa.av[i] + pa.aw[i]*pa.aw[i]
        )
    result = dt_max
    if max_cfl > 0.0:
        result = min(result, cfl * hmin / max_cfl)
    if max_force > 0.0:
        result = min(result, cfl * np.sqrt(hmin / np.sqrt(max_force)))
    return min(max(result, dt_min), dt_max)


def test_warp_isothermal_eos_matches_cpu_and_pulls_pressure():
    pa = get_particle_array(
        name='fluid',
        rho=[900.0, 1000.0, 1100.0],
        p=[0.0, 0.0, 0.0],
        backend='warp',
    )
    expected = 5.0 + 20.0*20.0*(pa.rho - 1000.0)

    actual = compute_isothermal_eos(pa, rho0=1000.0, c0=20.0,
                                    p0=5.0).get()
    pa.gpu.pull('p')

    assert np.allclose(actual, expected)
    assert np.allclose(pa.p, expected)


def test_warp_tait_eos_matches_cpu_and_pulls_pressure_and_sound_speed():
    rho = np.asarray([0.9, 1.0, 1.1, 1.25])
    pa = get_particle_array(
        name='fluid',
        rho=rho,
        p=np.zeros_like(rho),
        cs=np.zeros_like(rho),
        backend='warp',
    )
    expected_p, expected_cs = _cpu_tait_eos(
        rho, rho0=1.0, c0=20.0, gamma=7.0, p0=0.5
    )

    p, cs = compute_tait_eos(
        pa, rho0=1.0, c0=20.0, gamma=7.0, p0=0.5
    )
    pa.gpu.pull('p', 'cs')

    assert np.allclose(p.get(), expected_p)
    assert np.allclose(cs.get(), expected_cs)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.cs, expected_cs)


def test_warp_summation_density_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_summation_density(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    actual = compute_summation_density(nnps, 0, 0).get()

    assert np.allclose(actual, expected)


def test_warp_gaussian_summation_density_matches_pysph_kernel():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 0.62],
        y=[0.0, 0.0, 0.1, -0.05],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.3],
        m=[1.0, 2.0, 1.5, 1.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_summation_density(
        particles, 0, 0, dim=2, radius_scale=3.0, kernel='gaussian'
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=3.0)

    actual = compute_summation_density(
        nnps, 0, 0, kernel='gaussian'
    ).get()

    assert np.allclose(actual, expected)


def test_warp_summation_density_matches_cpu_cross_array_in_3d_and_pulls_rho():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_summation_density(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_summation_density(nnps, 0, 1)
    solid.gpu.pull('rho')

    assert np.allclose(solid.rho, expected)


def test_warp_continuity_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        u=[1.0, 0.5, -0.5, 0.0],
        v=[0.0, 0.2, 0.4, -0.1],
        w=[0.0, 0.0, 0.0, 0.0],
        arho=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_continuity(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    actual = compute_continuity(nnps, 0, 0).get()

    assert np.allclose(actual, expected)


def test_warp_continuity_matches_cpu_cross_array_in_3d_and_pulls_arho():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        u=[1.0, 0.5, -0.5, 0.0],
        v=[0.0, 0.2, 0.4, -0.1],
        w=[0.1, 0.3, -0.2, 0.0],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        u=[-0.1, 0.2],
        v=[0.3, -0.2],
        w=[0.0, 0.1],
        arho=[0.0, 0.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_continuity(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_continuity(nnps, 0, 1)
    solid.gpu.pull('arho')

    assert np.allclose(solid.arho, expected)


def test_warp_pressure_gradient_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        au=[0.0, 0.0, 0.0, 0.0],
        av=[0.0, 0.0, 0.0, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_pressure_gradient(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    au, av, aw = compute_pressure_gradient(nnps, 0, 0)

    assert np.allclose(au.get(), expected[:, 0])
    assert np.allclose(av.get(), expected[:, 1])
    assert np.allclose(aw.get(), expected[:, 2])


def test_warp_gaussian_pressure_gradient_matches_pysph_kernel():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 0.62],
        y=[0.0, 0.0, 0.1, -0.05],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.3],
        m=[1.0, 2.0, 1.5, 1.0],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        au=[0.0, 0.0, 0.0, 0.0],
        av=[0.0, 0.0, 0.0, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_pressure_gradient(
        particles, 0, 0, dim=2, radius_scale=3.0, kernel='gaussian'
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=3.0)

    au, av, aw = compute_pressure_gradient(
        nnps, 0, 0, kernel='gaussian'
    )

    assert np.allclose(au.get(), expected[:, 0])
    assert np.allclose(av.get(), expected[:, 1])
    assert np.allclose(aw.get(), expected[:, 2])


def test_warp_pressure_gradient_matches_cpu_cross_array_in_3d_and_pulls_accel():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        rho=[1.05, 0.95],
        p=[2.5, 1.0],
        au=[0.0, 0.0],
        av=[0.0, 0.0],
        aw=[0.0, 0.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_pressure_gradient(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_pressure_gradient(nnps, 0, 1)
    solid.gpu.pull('au', 'av', 'aw')

    assert np.allclose(solid.au, expected[:, 0])
    assert np.allclose(solid.av, expected[:, 1])
    assert np.allclose(solid.aw, expected[:, 2])


def test_warp_artificial_viscosity_matches_cpu_and_adds_to_acceleration():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.45, 1.2],
        y=[0.0, 0.03, -0.02, 0.1],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.35, 0.35, 0.4, 0.35],
        m=[1.0, 1.5, 1.2, 0.8],
        rho=[1.0, 1.1, 0.9, 1.2],
        cs=[4.0, 5.0, 6.0, 7.0],
        u=[1.0, -1.0, -0.2, 0.0],
        v=[0.0, 0.05, -0.1, 0.0],
        w=[0.0, 0.0, 0.0, 0.0],
        au=[0.5, -0.25, 0.1, 0.0],
        av=[0.0, 0.2, -0.1, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    alpha = 0.1
    beta = 0.2
    c0 = 5.0
    initial = np.column_stack([pa.au.copy(), pa.av.copy(), pa.aw.copy()])
    expected = initial + _cpu_artificial_viscosity(
        particles, 0, 0, dim=2, alpha=alpha, beta=beta, c0=c0
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    au, av, aw = compute_artificial_viscosity(
        nnps, 0, 0, alpha=alpha, beta=beta, c0=c0
    )

    assert np.allclose(au.get(), expected[:, 0])
    assert np.allclose(av.get(), expected[:, 1])
    assert np.allclose(aw.get(), expected[:, 2])


def test_warp_xsph_correction_matches_cpu_reference():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.45, 0.7],
        y=[0.0, 0.03, -0.02, 0.1],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.35, 0.35, 0.4, 0.35],
        m=[1.0, 1.5, 1.2, 0.8],
        rho=[1.0, 1.1, 0.9, 1.2],
        u=[1.0, -1.0, -0.2, 0.0],
        v=[0.0, 0.05, -0.1, 0.0],
        w=[0.0, 0.0, 0.0, 0.0],
        ax=[0.0, 0.0, 0.0, 0.0],
        ay=[0.0, 0.0, 0.0, 0.0],
        az=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    eps = 0.5
    expected = _cpu_xsph_correction(
        particles, 0, 0, dim=2, eps=eps, radius_scale=3.0,
        kernel='gaussian'
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=3.0)

    ax, ay, az = compute_xsph_correction(
        nnps, 0, 0, eps=eps, kernel='gaussian'
    )

    assert np.allclose(ax.get(), expected[:, 0])
    assert np.allclose(ay.get(), expected[:, 1])
    assert np.allclose(az.get(), expected[:, 2])


def test_warp_adaptive_timestep_matches_cpu_reference_and_clamps():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.45, 0.7],
        y=[0.0, 0.03, -0.02, 0.1],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.35, 0.35, 0.4, 0.35],
        m=[1.0, 1.5, 1.2, 0.8],
        u=[1.0, -1.0, -0.2, 0.0],
        v=[0.0, 0.05, -0.1, 0.0],
        w=[0.0, 0.0, 0.0, 0.0],
        au=[4.0, -0.5, 0.25, 0.0],
        av=[0.0, 0.2, -0.1, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    c0 = 5.0
    cfl = 0.3
    dt_min = 1.0e-6
    dt_max = 1.0e-2
    expected = _cpu_wcsph_dt(
        particles, 0, dim=2, c0=c0, cfl=cfl, dt_min=dt_min,
        dt_max=dt_max
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    actual = compute_wcsph_adaptive_timestep(
        nnps, 0, c0=c0, cfl=cfl, dt_min=dt_min, dt_max=dt_max
    )
    pa.gpu.pull('dt_cfl', 'dt_force')

    assert np.isclose(actual, expected)
    assert np.all(np.isfinite(pa.dt_cfl))
    assert np.all(np.isfinite(pa.dt_force))


def test_warp_equation_helpers_accept_prebuilt_neighbor_cache(monkeypatch):
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.45, 0.7],
        y=[0.0, 0.03, -0.02, 0.1],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.35, 0.35, 0.4, 0.35],
        m=[1.0, 1.5, 1.2, 0.8],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        u=[1.0, -1.0, -0.2, 0.0],
        v=[0.0, 0.05, -0.1, 0.0],
        w=[0.0, 0.0, 0.0, 0.0],
        au=[0.0, 0.0, 0.0, 0.0],
        av=[0.0, 0.0, 0.0, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        arho=[0.0, 0.0, 0.0, 0.0],
        ax=[0.0, 0.0, 0.0, 0.0],
        ay=[0.0, 0.0, 0.0, 0.0],
        az=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    cache = nnps.build_neighbor_cache_gpu(0, 0)
    calls = []
    original = nnps.build_neighbor_cache_gpu

    def counted_cache(src_index, dst_index):
        calls.append((src_index, dst_index))
        return original(src_index, dst_index)

    monkeypatch.setattr(nnps, 'build_neighbor_cache_gpu', counted_cache)

    compute_pressure_gradient(nnps, 0, 0)
    assert calls == [(0, 0)]

    compute_pressure_gradient(nnps, 0, 0, cache=cache)
    compute_artificial_viscosity(nnps, 0, 0, alpha=0.1, cache=cache)
    compute_continuity(nnps, 0, 0, cache=cache)
    compute_xsph_correction(nnps, 0, 0, eps=0.5, cache=cache)
    assert calls == [(0, 0)]


def test_warp_continuity_step_builds_no_flat_neighbor_cache(monkeypatch):
    # ADR-0004: both neighbor consumers (fused equations + adaptive CFL
    # dt-factors) walk the cell list directly, so the continuity step never
    # materializes a flat CSR neighbor list. adaptive_dt=True exercises both
    # consumers; the grid itself is still built (via _build_grid).
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    pa = get_particle_array(
        name='fluid',
        x=x.copy(), y=y.copy(), z=z.copy(),
        h=np.asarray([0.35, 0.35, 0.4, 0.35]),
        m=np.asarray([1.0, 1.5, 1.2, 0.8]),
        rho=np.asarray([1.0, 1.03, 0.98, 1.01]),
        p=np.zeros_like(x),
        cs=np.ones_like(x) * 5.0,
        u=np.asarray([0.1, -0.05, 0.2, 0.0]),
        v=np.asarray([0.0, 0.15, -0.1, 0.05]),
        w=z.copy(),
        au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
        ax=np.zeros_like(x), ay=np.zeros_like(x), az=np.zeros_like(x),
        arho=np.zeros_like(x),
        backend='warp',
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    original = nnps.build_neighbor_cache_gpu
    grid_original = nnps._build_grid
    calls = []
    grid_calls = []

    def counted_cache(src_index, dst_index):
        calls.append((src_index, dst_index))
        return original(src_index, dst_index)

    def counted_grid(src_index):
        grid_calls.append(src_index)
        return grid_original(src_index)

    monkeypatch.setattr(nnps, 'build_neighbor_cache_gpu', counted_cache)
    monkeypatch.setattr(nnps, '_build_grid', counted_grid)

    wc_sph_leapfrog_step(
        nnps, dt=1.0e-3, rho0=1.0, c0=5.0, alpha=0.1, beta=0.0,
        eos='tait', gamma=7.0, xsph_eps=0.5, adaptive_dt=True,
        cfl=0.3, dt_min=1.0e-8, dt_max=1.0e-2,
        density_mode='continuity'
    )

    # No flat neighbor cache anywhere on the continuity path, and the grid is
    # consulted (built once per half-stage, reused by both consumers).
    assert calls == []
    assert len(grid_calls) >= 2


def test_warp_equation_helpers_reject_custom_output_names():
    # The generator-backed helpers write the block's canonical output arrays;
    # a non-default out_prop/out_props can no longer be honored, so it must
    # fail fast rather than silently writing the wrong array. The guard runs
    # before any device work, so nnps is never dereferenced here.
    with pytest.raises(ValueError):
        compute_summation_density(None, out_prop='rho_custom')
    with pytest.raises(ValueError):
        compute_continuity(None, out_prop='arho_custom')
    with pytest.raises(ValueError):
        compute_pressure_gradient(None, out_props=('bu', 'bv', 'bw'))
    with pytest.raises(ValueError):
        compute_artificial_viscosity(None, out_props=('bu', 'bv', 'bw'))
    with pytest.raises(ValueError):
        compute_xsph_correction(None, out_props=('bx', 'by', 'bz'))


def test_warp_fused_accel_matches_separate_helpers():
    # Fusion-consistency check: after the generator migration both sides are
    # generator-backed (the fused 4-equation group vs the four single-block
    # helpers), so this asserts that fusing N blocks equals running them
    # separately and composing (pressure gradient overwrite, viscosity add).
    def make_pa():
        x = np.asarray([0.0, 0.2, 0.45, 0.7, 1.1])
        y = np.asarray([0.0, 0.03, -0.02, 0.1, -0.15])
        z = np.zeros_like(x)
        return get_particle_array(
            name='fluid', x=x.copy(), y=y.copy(), z=z.copy(),
            h=np.asarray([0.35, 0.35, 0.4, 0.35, 0.38]),
            m=np.asarray([1.0, 1.5, 1.2, 0.8, 1.1]),
            rho=np.asarray([1.0, 1.1, 0.9, 1.2, 1.05]),
            p=np.asarray([2.0, 3.0, 1.5, 0.5, 1.2]),
            cs=np.asarray([5.0, 5.0, 5.0, 5.0, 5.0]),
            u=np.asarray([1.0, -1.0, -0.2, 0.0, 0.3]),
            v=np.asarray([0.0, 0.05, -0.1, 0.0, 0.2]),
            w=z.copy(),
            au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
            arho=np.zeros_like(x), ax=np.zeros_like(x), ay=np.zeros_like(x),
            az=np.zeros_like(x), backend='warp',
        )

    alpha, beta, eps, kernel = 0.15, 0.05, 0.5, 'gaussian'

    # Reference: the four separate helpers chained on one cache. Inputs are
    # pushed once, then helpers run with push=False so artificial viscosity
    # accumulates onto the pressure-gradient result instead of clobbering it.
    pa_sep = make_pa()
    nnps_sep = UniformGridWarpNNPS(dim=2, particles=[pa_sep], radius_scale=3.0)
    pa_sep.gpu.push(
        'x', 'y', 'z', 'h', 'm', 'rho', 'p', 'cs', 'u', 'v', 'w',
        'au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'
    )
    cache_sep = nnps_sep.build_neighbor_cache_gpu(0, 0)
    compute_pressure_gradient(
        nnps_sep, 0, 0, kernel=kernel, cache=cache_sep, push=False
    )
    compute_artificial_viscosity(
        nnps_sep, 0, 0, alpha=alpha, beta=beta, kernel=kernel,
        cache=cache_sep, push=False
    )
    compute_continuity(
        nnps_sep, 0, 0, kernel=kernel, cache=cache_sep, push=False
    )
    compute_xsph_correction(
        nnps_sep, 0, 0, eps=eps, kernel=kernel, cache=cache_sep, push=False
    )
    pa_sep.gpu.pull('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')

    # Fused: one generated group kernel.
    pa_fused = make_pa()
    nnps_fused = UniformGridWarpNNPS(
        dim=2, particles=[pa_fused], radius_scale=3.0
    )
    compute_wcsph_accel_continuity(
        nnps_fused, 0, 0, alpha=alpha, beta=beta, eps=eps, kernel=kernel,
        push=True
    )
    pa_fused.gpu.pull('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')

    for name in ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'):
        assert np.allclose(
            getattr(pa_sep, name), getattr(pa_fused, name),
            rtol=1e-5, atol=1e-6
        ), name


def test_warp_grid_direct_accel_matches_flat_fused():
    # ADR-0004: the grid-direct fused kernel must visit exactly the neighbor
    # set the flat CSR list contained (same support cutoff), so its result
    # matches the flat fused kernel to fp32 reordering scale. This isolates the
    # neighbor-mode change from the equation math.
    def make_pa():
        x = np.asarray([0.0, 0.2, 0.45, 0.7, 1.1, 0.15, 0.9])
        y = np.asarray([0.0, 0.03, -0.02, 0.1, -0.15, 0.25, 0.18])
        z = np.zeros_like(x)
        return get_particle_array(
            name='fluid', x=x.copy(), y=y.copy(), z=z.copy(),
            h=np.asarray([0.35, 0.35, 0.4, 0.35, 0.38, 0.36, 0.34]),
            m=np.asarray([1.0, 1.5, 1.2, 0.8, 1.1, 0.95, 1.05]),
            rho=np.asarray([1.0, 1.1, 0.9, 1.2, 1.05, 0.97, 1.03]),
            p=np.asarray([2.0, 3.0, 1.5, 0.5, 1.2, 0.8, 1.7]),
            cs=np.ones_like(x) * 5.0,
            u=np.asarray([1.0, -1.0, -0.2, 0.0, 0.3, 0.4, -0.3]),
            v=np.asarray([0.0, 0.05, -0.1, 0.0, 0.2, -0.15, 0.1]),
            w=z.copy(),
            au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
            arho=np.zeros_like(x), ax=np.zeros_like(x), ay=np.zeros_like(x),
            az=np.zeros_like(x), backend='warp',
        )

    alpha, beta, eps, kernel = 0.15, 0.05, 0.5, 'gaussian'

    pa_flat = make_pa()
    nnps_flat = UniformGridWarpNNPS(dim=2, particles=[pa_flat], radius_scale=3.0)
    compute_wcsph_accel_continuity(
        nnps_flat, 0, 0, alpha=alpha, beta=beta, eps=eps, kernel=kernel,
        push=True, neighbor_mode='flat'
    )
    pa_flat.gpu.pull('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')

    pa_grid = make_pa()
    nnps_grid = UniformGridWarpNNPS(dim=2, particles=[pa_grid], radius_scale=3.0)
    compute_wcsph_accel_continuity(
        nnps_grid, 0, 0, alpha=alpha, beta=beta, eps=eps, kernel=kernel,
        push=True, neighbor_mode='grid'
    )
    pa_grid.gpu.pull('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')

    for name in ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'):
        assert np.allclose(
            getattr(pa_flat, name), getattr(pa_grid, name),
            rtol=1e-5, atol=1e-6
        ), name


def test_warp_continuity_step_issues_single_fused_launch_per_stage(
        monkeypatch):
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    pa = get_particle_array(
        name='fluid', x=x.copy(), y=y.copy(), z=z.copy(),
        h=np.asarray([0.35, 0.35, 0.4, 0.35]),
        m=np.asarray([1.0, 1.5, 1.2, 0.8]),
        rho=np.asarray([1.0, 1.03, 0.98, 1.01]),
        p=np.zeros_like(x), cs=np.ones_like(x) * 5.0,
        u=np.asarray([0.1, -0.05, 0.2, 0.0]),
        v=np.asarray([0.0, 0.15, -0.1, 0.05]), w=z.copy(),
        au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
        ax=np.zeros_like(x), ay=np.zeros_like(x), az=np.zeros_like(x),
        arho=np.zeros_like(x), backend='warp',
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)

    counts = {'fused': 0, 'separate': 0}
    fused_original = warp_sph.compute_wcsph_accel_continuity

    def counted_fused(*args, **kwargs):
        counts['fused'] += 1
        return fused_original(*args, **kwargs)

    def counted_separate(name):
        original = getattr(warp_sph, name)

        def wrapper(*args, **kwargs):
            counts['separate'] += 1
            return original(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(
        warp_sph, 'compute_wcsph_accel_continuity', counted_fused
    )
    for name in ('compute_pressure_gradient', 'compute_artificial_viscosity',
                 'compute_continuity', 'compute_xsph_correction'):
        monkeypatch.setattr(warp_sph, name, counted_separate(name))

    wc_sph_leapfrog_step(
        nnps, dt=1.0e-3, rho0=1.0, c0=5.0, alpha=0.1, beta=0.0,
        eos='tait', gamma=7.0, xsph_eps=0.5, density_mode='continuity'
    )

    # One fused launch per PEC half-stage, and none of the per-equation
    # neighbor-loop helpers are called in the continuity path.
    assert counts['fused'] == 2
    assert counts['separate'] == 0


def test_warp_leapfrog_adaptive_timestep_scale_and_step_cap():
    def make_pa():
        x = np.asarray([0.0, 0.2, 0.45, 0.7])
        y = np.asarray([0.0, 0.03, -0.02, 0.1])
        z = np.zeros_like(x)
        return get_particle_array(
            name='fluid',
            x=x.copy(), y=y.copy(), z=z.copy(),
            h=np.asarray([0.35, 0.35, 0.4, 0.35]),
            m=np.asarray([1.0, 1.5, 1.2, 0.8]),
            rho=np.ones_like(x),
            p=np.zeros_like(x),
            cs=np.ones_like(x) * 5.0,
            u=np.asarray([1.0, -1.0, -0.2, 0.0]),
            v=np.asarray([0.0, 0.05, -0.1, 0.0]),
            w=z.copy(),
            au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
            backend='warp',
        )

    raw_pa = make_pa()
    raw_nnps = UniformGridWarpNNPS(
        dim=2, particles=[raw_pa], radius_scale=2.0
    )
    _result, raw_dt = wc_sph_leapfrog_step(
        raw_nnps, dt=1.0e-3, rho0=1.0, c0=5.0, adaptive_dt=True,
        cfl=0.3, dt_min=1.0e-8, dt_max=1.0e-2, return_dt=True
    )

    scaled_pa = make_pa()
    scaled_nnps = UniformGridWarpNNPS(
        dim=2, particles=[scaled_pa], radius_scale=2.0
    )
    _result, scaled_dt = wc_sph_leapfrog_step(
        scaled_nnps, dt=1.0e-3, rho0=1.0, c0=5.0, adaptive_dt=True,
        cfl=0.3, dt_min=1.0e-8, dt_max=1.0e-2, return_dt=True,
        adaptive_dt_scale=0.5
    )

    capped_pa = make_pa()
    capped_nnps = UniformGridWarpNNPS(
        dim=2, particles=[capped_pa], radius_scale=2.0
    )
    _result, capped_dt = wc_sph_leapfrog_step(
        capped_nnps, dt=1.0e-3, rho0=1.0, c0=5.0, adaptive_dt=True,
        cfl=0.3, dt_min=1.0e-8, dt_max=1.0e-2, return_dt=True,
        step_dt_max=1.0e-7
    )

    assert np.isclose(scaled_dt, 0.5 * raw_dt)
    assert np.isclose(capped_dt, 1.0e-7)


def test_warp_wc_sph_euler_step_with_tait_eos_uses_sound_speed_in_avisc():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.5, -0.4, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    gamma = 7.0
    p0 = 0.1
    alpha = 0.1
    beta = 0.2
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        cs=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    expected_rho = _cpu_summation_density(particles, 0, 0, dim=2)
    expected_p, expected_cs = _cpu_tait_eos(
        expected_rho, rho0=rho0, c0=c0, gamma=gamma, p0=p0
    )
    expected_pa = get_particle_array(
        name='expected',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=expected_rho,
        p=expected_p,
        cs=expected_cs,
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        backend='warp',
    )
    expected_acc = (
        _cpu_pressure_gradient([expected_pa], 0, 0, dim=2) +
        _cpu_artificial_viscosity(
            [expected_pa], 0, 0, dim=2, alpha=alpha, beta=beta, c0=c0
        )
    )
    expected_u = u + dt*expected_acc[:, 0]
    expected_v = v + dt*expected_acc[:, 1]
    expected_w = w + dt*expected_acc[:, 2]

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_euler_step(
        nnps, dt=dt, rho0=rho0, c0=c0, p0=p0, alpha=alpha, beta=beta,
        eos='tait', gamma=gamma
    )
    pa.gpu.pull('rho', 'p', 'cs', 'au', 'av', 'aw', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.cs))
    assert np.allclose(pa.rho, expected_rho)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.cs, expected_cs)
    assert np.allclose(pa.au, expected_acc[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, expected_acc[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, expected_acc[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)


def test_warp_euler_step_updates_velocity_and_position_on_device():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0, 2.0],
        y=[0.5, 1.5, 2.5],
        z=[1.0, 2.0, 3.0],
        u=[1.0, -1.0, 0.5],
        v=[0.0, 2.0, -0.5],
        w=[0.25, -0.25, 1.0],
        au=[0.1, 0.2, -0.3],
        av=[-0.2, 0.4, 0.1],
        aw=[0.5, -0.5, 0.25],
        backend='warp',
    )
    dt = 0.25
    old_x = pa.x.copy()
    old_y = pa.y.copy()
    old_z = pa.z.copy()
    old_u = pa.u.copy()
    old_v = pa.v.copy()
    old_w = pa.w.copy()
    au = pa.au.copy()
    av = pa.av.copy()
    aw = pa.aw.copy()

    euler_step(pa, dt=dt, dim=2)
    pa.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')

    expected_u = old_u + dt*au
    expected_v = old_v + dt*av
    expected_w = old_w + dt*aw
    assert np.allclose(pa.u, expected_u)
    assert np.allclose(pa.v, expected_v)
    assert np.allclose(pa.w, expected_w)
    assert np.allclose(pa.x, old_x + dt*expected_u)
    assert np.allclose(pa.y, old_y + dt*expected_v)
    assert np.allclose(pa.z, old_z)


def test_warp_leapfrog_kick_drift_and_wrap_update_device_state():
    pa = get_particle_array(
        name='fluid',
        x=[0.95, -0.2, 1.8],
        y=[0.9, 1.2, -0.1],
        z=[0.0, 0.0, 0.0],
        u=[0.4, 0.5, -0.25],
        v=[0.3, -0.4, 0.1],
        w=[0.0, 0.0, 0.0],
        au=[0.2, -0.1, 0.4],
        av=[-0.3, 0.2, 0.0],
        aw=[0.0, 0.0, 0.0],
        backend='warp',
    )
    dt = 0.5
    expected_u = pa.u + 0.5*dt*pa.au
    expected_v = pa.v + 0.5*dt*pa.av
    expected_x = pa.x + dt*expected_u
    expected_y = pa.y + dt*expected_v
    expected_x = expected_x - np.floor(expected_x)
    expected_y = expected_y - np.floor(expected_y)

    leapfrog_kick(pa, dt=0.5*dt, dim=2)
    leapfrog_drift(pa, dt=dt, dim=2, push=False)
    wrap_periodic(
        pa,
        {'xmin': 0.0, 'xmax': 1.0, 'ymin': 0.0, 'ymax': 1.0},
        dim=2,
    )
    pa.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')

    assert np.allclose(pa.u, expected_u)
    assert np.allclose(pa.v, expected_v)
    assert np.allclose(pa.x, expected_x)
    assert np.allclose(pa.y, expected_y)
    assert np.allclose(pa.z, np.zeros(3))


def test_warp_wc_sph_euler_step_matches_cpu_expected_state():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.1, -0.05, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    p0 = 0.1
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    expected_rho = _cpu_summation_density(particles, 0, 0, dim=2)
    expected_p = p0 + c0*c0*(expected_rho - rho0)
    expected_pa = get_particle_array(
        name='expected',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=expected_rho,
        p=expected_p,
        backend='warp',
    )
    expected_acc = _cpu_pressure_gradient([expected_pa], 0, 0, dim=2)
    expected_u = u + dt*expected_acc[:, 0]
    expected_v = v + dt*expected_acc[:, 1]
    expected_w = w + dt*expected_acc[:, 2]
    expected_x = x + dt*expected_u
    expected_y = y + dt*expected_v

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_euler_step(nnps, dt=dt, rho0=rho0, c0=c0, p0=p0)
    pa.gpu.pull('rho', 'p', 'au', 'av', 'aw', 'x', 'y', 'z', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.rho))
    assert np.all(np.isfinite(pa.p))
    assert np.all(np.isfinite(pa.au))
    assert np.allclose(pa.rho, expected_rho)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.au, expected_acc[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, expected_acc[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, expected_acc[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.x, expected_x, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.y, expected_y, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.z, z)


def test_warp_wc_sph_leapfrog_step_matches_cpu_expected_state():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.1, -0.05, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    p0 = 0.1
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    rho_n = _cpu_summation_density(particles, 0, 0, dim=2)
    p_n = p0 + c0*c0*(rho_n - rho0)
    force_pa_n = get_particle_array(
        name='expected_n',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=rho_n,
        p=p_n,
        backend='warp',
    )
    acc_n = _cpu_pressure_gradient([force_pa_n], 0, 0, dim=2)
    u_half = u + 0.5*dt*acc_n[:, 0]
    v_half = v + 0.5*dt*acc_n[:, 1]
    w_half = w + 0.5*dt*acc_n[:, 2]
    x_np1 = x + dt*u_half
    y_np1 = y + dt*v_half
    z_np1 = z + dt*w_half

    force_pa_np1 = get_particle_array(
        name='expected_np1',
        x=x_np1.copy(),
        y=y_np1.copy(),
        z=z_np1.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        backend='warp',
    )
    rho_np1 = _cpu_summation_density([force_pa_np1], 0, 0, dim=2)
    p_np1 = p0 + c0*c0*(rho_np1 - rho0)
    force_pa_np1.rho[:] = rho_np1
    force_pa_np1.p[:] = p_np1
    acc_np1 = _cpu_pressure_gradient([force_pa_np1], 0, 0, dim=2)
    expected_u = u_half + 0.5*dt*acc_np1[:, 0]
    expected_v = v_half + 0.5*dt*acc_np1[:, 1]
    expected_w = w_half + 0.5*dt*acc_np1[:, 2]

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_leapfrog_step(nnps, dt=dt, rho0=rho0, c0=c0, p0=p0)
    pa.gpu.pull('rho', 'p', 'au', 'av', 'aw', 'x', 'y', 'z', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.rho))
    assert np.all(np.isfinite(pa.p))
    assert np.all(np.isfinite(pa.au))
    assert np.allclose(pa.rho, rho_np1)
    assert np.allclose(pa.p, p_np1)
    assert np.allclose(pa.au, acc_np1[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, acc_np1[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, acc_np1[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.x, x_np1, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.y, y_np1, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.z, z_np1, rtol=1e-5, atol=1e-5)


def test_warp_wcsph_pec_stage_uses_saved_density_and_xsph_advection():
    x = np.asarray([0.0, 0.2, 0.45])
    y = np.asarray([0.0, 0.1, -0.05])
    z = np.zeros_like(x)
    rho = np.asarray([1.0, 1.1, 0.95])
    u = np.asarray([0.1, -0.05, 0.2])
    v = np.asarray([0.0, 0.15, -0.1])
    w = np.zeros_like(x)
    au = np.asarray([0.3, -0.2, 0.05])
    av = np.asarray([-0.1, 0.25, 0.4])
    aw = np.zeros_like(x)
    ax = np.asarray([0.02, -0.01, 0.03])
    ay = np.asarray([0.0, 0.015, -0.025])
    az = np.zeros_like(x)
    arho = np.asarray([0.4, -0.2, 0.1])
    dt = 0.01
    stage = 0.5
    pa = get_particle_array(
        name='fluid',
        x=x.copy(), y=y.copy(), z=z.copy(),
        u=u.copy(), v=v.copy(), w=w.copy(), rho=rho.copy(),
        au=au.copy(), av=av.copy(), aw=aw.copy(),
        ax=ax.copy(), ay=ay.copy(), az=az.copy(), arho=arho.copy(),
        backend='warp',
    )

    save_wcsph_state(pa, dim=2)
    wcsph_pec_stage(pa, dt=dt, stage=stage, dim=2, xsph=True, push=False)
    pa.gpu.pull('x', 'y', 'rho', 'u', 'v')

    dt_factor = dt * stage
    assert np.allclose(pa.u, u + dt_factor * au)
    assert np.allclose(pa.v, v + dt_factor * av)
    assert np.allclose(pa.rho, rho + dt_factor * arho)
    assert np.allclose(pa.x, x + dt_factor * (u + ax))
    assert np.allclose(pa.y, y + dt_factor * (v + ay))


def test_warp_wc_sph_leapfrog_continuity_mode_matches_cpu_pec_state():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    rho = np.asarray([1.0, 1.03, 0.98, 1.01])
    u = np.asarray([0.1, -0.05, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    gamma = 7.0
    xsph_eps = 0.5
    pa = get_particle_array(
        name='fluid',
        x=x.copy(), y=y.copy(), z=z.copy(), h=h.copy(), m=m.copy(),
        rho=rho.copy(), p=np.zeros_like(x), cs=np.ones_like(x)*c0,
        u=u.copy(), v=v.copy(), w=w.copy(),
        au=np.zeros_like(x), av=np.zeros_like(x), aw=np.zeros_like(x),
        ax=np.zeros_like(x), ay=np.zeros_like(x), az=np.zeros_like(x),
        arho=np.zeros_like(x), backend='warp',
    )
    p_n, cs_n = _cpu_tait_eos(rho, rho0=rho0, c0=c0, gamma=gamma)
    cpu_n = get_particle_array(
        name='cpu_n',
        x=x.copy(), y=y.copy(), z=z.copy(), h=h.copy(), m=m.copy(),
        rho=rho.copy(), p=p_n, cs=cs_n,
        u=u.copy(), v=v.copy(), w=w.copy(), backend='warp',
    )
    particles_n = [cpu_n]
    acc_n = _cpu_pressure_gradient(particles_n, 0, 0, dim=2)
    arho_n = _cpu_continuity(particles_n, 0, 0, dim=2)
    xsph_n = _cpu_xsph_correction(
        particles_n, 0, 0, dim=2, eps=xsph_eps
    )
    u_half = u + 0.5 * dt * acc_n[:, 0]
    v_half = v + 0.5 * dt * acc_n[:, 1]
    w_half = w + 0.5 * dt * acc_n[:, 2]
    rho_half = rho + 0.5 * dt * arho_n
    x_half = x + 0.5 * dt * (u + xsph_n[:, 0])
    y_half = y + 0.5 * dt * (v + xsph_n[:, 1])
    z_half = z + 0.5 * dt * (w + xsph_n[:, 2])

    p_half, cs_half = _cpu_tait_eos(
        rho_half, rho0=rho0, c0=c0, gamma=gamma
    )
    cpu_half = get_particle_array(
        name='cpu_half',
        x=x_half.copy(), y=y_half.copy(), z=z_half.copy(),
        h=h.copy(), m=m.copy(), rho=rho_half.copy(), p=p_half,
        cs=cs_half, u=u_half.copy(), v=v_half.copy(), w=w_half.copy(),
        backend='warp',
    )
    particles_half = [cpu_half]
    acc_half = _cpu_pressure_gradient(particles_half, 0, 0, dim=2)
    arho_half = _cpu_continuity(particles_half, 0, 0, dim=2)
    xsph_half = _cpu_xsph_correction(
        particles_half, 0, 0, dim=2, eps=xsph_eps
    )
    expected_u = u + dt * acc_half[:, 0]
    expected_v = v + dt * acc_half[:, 1]
    expected_w = w + dt * acc_half[:, 2]
    expected_rho = rho + dt * arho_half
    expected_x = x + dt * (u_half + xsph_half[:, 0])
    expected_y = y + dt * (v_half + xsph_half[:, 1])
    expected_z = z + dt * (w_half + xsph_half[:, 2])

    nnps = UniformGridWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    wc_sph_leapfrog_step(
        nnps, dt=dt, rho0=rho0, c0=c0, gamma=gamma, eos='tait',
        xsph_eps=xsph_eps, density_mode='continuity'
    )
    pa.gpu.pull('x', 'y', 'z', 'rho', 'u', 'v', 'w', 'au', 'av', 'aw', 'arho')

    assert np.allclose(pa.x, expected_x, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.y, expected_y, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.z, expected_z, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.rho, expected_rho, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.au, acc_half[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, acc_half[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, acc_half[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.arho, arho_half, rtol=1e-5, atol=1e-5)
