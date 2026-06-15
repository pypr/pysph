import numpy as np
import pytest

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

pytest.importorskip('warp')

from cyarray.carray import UIntArray

from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import BruteForceWarpNNPS, UniformGridWarpNNPS


def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return np.sort(nbrs.get_npy_array()[:nbrs.length])


def _assert_all_neighbors_match(cpu, warp, particles, pairs):
    for src_index, dst_index in pairs:
        warp.set_context(src_index, dst_index)
        dst_count = particles[dst_index].get_number_of_particles()
        for d_idx in range(dst_count):
            expected = _neighbors(cpu, src_index, dst_index, d_idx)
            actual = _neighbors(warp, src_index, dst_index, d_idx)
            assert np.array_equal(actual, expected)


def _neighbor_sum(cpu, particles, src_index, dst_index, prop):
    values = particles[src_index].properties[prop].get_npy_array()
    dst_count = particles[dst_index].get_number_of_particles()
    result = np.zeros(dst_count, dtype=values.dtype)
    for d_idx in range(dst_count):
        result[d_idx] = np.sum(values[_neighbors(cpu, src_index, dst_index,
                                                 d_idx)])
    return result


def test_brute_force_warp_nnps_matches_cpu_linked_list_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.25, 0.25],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    warp = BruteForceWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    _assert_all_neighbors_match(cpu, warp, particles, [(0, 0)])


def test_brute_force_warp_nnps_matches_cpu_for_multiple_arrays():
    pa1 = get_particle_array(
        name='fluid',
        x=[0.0, 0.25, 0.5],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[0.2, 0.2, 0.2],
        backend='warp',
    )
    pa2 = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        backend='warp',
    )
    particles = [pa1, pa2]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    warp = BruteForceWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    _assert_all_neighbors_match(
        cpu, warp, particles, [(0, 0), (1, 1), (0, 1), (1, 0)]
    )


def test_brute_force_warp_nnps_uses_source_and_destination_h():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[0.1, 1.1, 0.1],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=1, particles=particles, radius_scale=1.0)
    warp = BruteForceWarpNNPS(dim=1, particles=particles, radius_scale=1.0)

    _assert_all_neighbors_match(cpu, warp, particles, [(0, 0)])


def test_brute_force_warp_nnps_can_sort_neighbors_by_gid():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.1, 0.2],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[1.0, 1.0, 1.0],
        gid=np.array([30, 10, 20], dtype=np.uint32),
        backend='warp',
    )
    warp = BruteForceWarpNNPS(
        dim=1, particles=[pa], radius_scale=1.0, sort_gids=True
    )

    nbrs = UIntArray()
    warp.get_nearest_particles(0, 0, 0, nbrs)

    assert np.array_equal(nbrs.get_npy_array()[:nbrs.length],
                          np.array([1, 2, 0], dtype=np.uint32))


def test_brute_force_warp_nnps_update_after_particle_mutation():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        backend='warp',
    )
    particles = [pa]
    warp = BruteForceWarpNNPS(dim=1, particles=particles, radius_scale=1.0)

    assert np.array_equal(_neighbors(warp, 0, 0, 0),
                          np.array([0], dtype=np.uint32))

    pa.x[1] = 0.1
    warp.update()

    assert np.array_equal(_neighbors(warp, 0, 0, 0),
                          np.array([0, 1], dtype=np.uint32))


def test_cached_brute_force_warp_nnps_matches_uncached_path():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.25, 0.25],
        backend='warp',
    )
    particles = [pa]
    cached = BruteForceWarpNNPS(
        dim=2, particles=particles, radius_scale=2.0, cache=True
    )
    uncached = BruteForceWarpNNPS(
        dim=2, particles=particles, radius_scale=2.0, cache=False
    )

    _assert_all_neighbors_match(uncached, cached, particles, [(0, 0)])


def test_cached_brute_force_warp_nnps_rebuilds_after_update():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        backend='warp',
    )
    warp = BruteForceWarpNNPS(
        dim=1, particles=[pa], radius_scale=1.0, cache=True
    )

    assert np.array_equal(_neighbors(warp, 0, 0, 0),
                          np.array([0], dtype=np.uint32))

    pa.x[1] = 0.1
    warp.update()

    assert np.array_equal(_neighbors(warp, 0, 0, 0),
                          np.array([0, 1], dtype=np.uint32))


def test_uniform_grid_warp_nnps_matches_cpu_linked_list_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.25, 0.25],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    _assert_all_neighbors_match(cpu, grid, particles, [(0, 0)])


def test_uniform_grid_warp_nnps_matches_cpu_for_multiple_arrays():
    pa1 = get_particle_array(
        name='fluid',
        x=[0.0, 0.25, 0.5],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[0.2, 0.2, 0.2],
        backend='warp',
    )
    pa2 = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        backend='warp',
    )
    particles = [pa1, pa2]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    _assert_all_neighbors_match(
        cpu, grid, particles, [(0, 0), (1, 1), (0, 1), (1, 0)]
    )


def test_uniform_grid_warp_nnps_matches_cpu_in_3d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5, 0.25],
        y=[0.0, 0.0, 0.1, 1.5, 0.25],
        z=[0.0, 0.1, 0.0, 1.5, 0.2],
        h=[0.25, 0.25, 0.25, 0.25, 0.25],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=3, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    _assert_all_neighbors_match(cpu, grid, particles, [(0, 0)])


def test_uniform_grid_warp_nnps_matches_bruteforce_for_variable_h():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[0.1, 1.1, 0.1],
        backend='warp',
    )
    particles = [pa]
    brute = BruteForceWarpNNPS(
        dim=1, particles=particles, radius_scale=1.0, cache=True
    )
    grid = UniformGridWarpNNPS(dim=1, particles=particles, radius_scale=1.0)

    _assert_all_neighbors_match(brute, grid, particles, [(0, 0)])


def test_uniform_grid_warp_nnps_rebuilds_after_update():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        backend='warp',
    )
    grid = UniformGridWarpNNPS(dim=1, particles=[pa], radius_scale=1.0)

    assert np.array_equal(_neighbors(grid, 0, 0, 0),
                          np.array([0], dtype=np.uint32))

    pa.x[1] = 0.1
    grid.update()

    assert np.array_equal(_neighbors(grid, 0, 0, 0),
                          np.array([0, 1], dtype=np.uint32))


def test_uniform_grid_warp_nnps_computes_neighbor_sum_on_device():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.25, 0.25],
        m=[1.0, 2.0, 3.0, 4.0],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    expected = _neighbor_sum(cpu, particles, 0, 0, 'm')
    actual = grid.compute_neighbor_sum(0, 0, 'm').numpy()

    assert np.allclose(actual, expected)


def test_uniform_grid_warp_nnps_computes_cross_array_neighbor_sum_on_device():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.25, 0.5],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        h=[0.2, 0.2, 0.2],
        m=[2.0, 4.0, 8.0],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        m=[1.0, 1.0],
        backend='warp',
    )
    particles = [fluid, solid]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    expected = _neighbor_sum(cpu, particles, 0, 1, 'm')
    actual = grid.compute_neighbor_sum(0, 1, 'm').numpy()

    assert np.allclose(actual, expected)


def test_uniform_grid_warp_nnps_neighbor_sum_rebuilds_after_update():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.2, 0.2],
        m=[2.0, 3.0],
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=1, particles=particles, radius_scale=1.0)
    grid = UniformGridWarpNNPS(dim=1, particles=particles, radius_scale=1.0)

    expected = _neighbor_sum(cpu, particles, 0, 0, 'm')
    assert np.allclose(grid.compute_neighbor_sum(0, 0, 'm').numpy(),
                       expected)

    pa.x[1] = 0.1
    cpu.update()
    grid.update()

    expected = _neighbor_sum(cpu, particles, 0, 0, 'm')
    assert np.allclose(grid.compute_neighbor_sum(0, 0, 'm').numpy(),
                       expected)
