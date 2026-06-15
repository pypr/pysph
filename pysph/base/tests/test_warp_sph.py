import numpy as np
import pytest

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

pytest.importorskip('warp')

from cyarray.carray import UIntArray

from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import (
    compute_continuity, compute_isothermal_eos, compute_pressure_gradient,
    compute_summation_density
)


def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return nbrs.get_npy_array()[:nbrs.length]


def _cpu_summation_density(particles, src_index, dst_index, dim,
                           radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
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
            total += src.m[s_idx] * kernel.kernel(xij=xij, rij=rij, h=hij)
        result[d_idx] = total
    return result


def _cpu_continuity(particles, src_index, dst_index, dim, radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
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
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            total += src.m[s_idx] * (
                vij[0]*dwij[0] + vij[1]*dwij[1] + vij[2]*dwij[2]
            )
        result[d_idx] = total
    return result


def _cpu_pressure_gradient(particles, src_index, dst_index, dim,
                           radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
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
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            rhoj21 = 1.0/(src.rho[s_idx]*src.rho[s_idx])
            tmp = tmpi + src.p[s_idx]*rhoj21
            acc += -src.m[s_idx] * tmp * np.asarray(dwij)
        result[d_idx, :] = acc
    return result


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
