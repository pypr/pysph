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
from pysph.base.warp_sph import compute_summation_density


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
