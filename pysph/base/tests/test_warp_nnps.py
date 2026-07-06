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
from pysph.base.warp_nnps import (
    BruteForceWarpNNPS, UniformGridWarpNNPS,
    assign_particle_levels, brute_force_neighbor_sets,
    accepted_level_pair_counts,
)
from pysph.base.warp_multilevel_nnps import MultilevelGridWarpNNPS


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


def _assert_device_cache_neighbors_match(cpu, grid, particles, pairs):
    for src_index, dst_index in pairs:
        cache = grid.build_neighbor_cache_gpu(src_index, dst_index)
        starts = cache['starts_dev'].numpy()
        lengths = cache['lengths']
        neighbors = cache['neighbors_dev'].numpy()
        dst_count = particles[dst_index].get_number_of_particles()

        assert len(starts) == dst_count
        assert len(lengths) == dst_count
        assert len(neighbors) == cache['total_neighbors']

        for d_idx in range(dst_count):
            expected = _neighbors(cpu, src_index, dst_index, d_idx)
            start = int(starts[d_idx])
            stop = start + int(lengths[d_idx])
            actual = np.sort(neighbors[start:stop])
            assert np.array_equal(actual, expected), (
                src_index, dst_index, d_idx, actual, expected
            )


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


def test_uniform_grid_warp_nnps_device_cache_matches_cpu_indices_in_random_2d():
    rng = np.random.default_rng(1729)
    n = 96
    pa = get_particle_array(
        name='fluid',
        x=rng.random(n),
        y=rng.random(n),
        z=np.zeros(n),
        h=0.055 + 0.035 * rng.random(n),
        backend='warp',
    )
    particles = [pa]
    cpu = LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    _assert_device_cache_neighbors_match(cpu, grid, particles, [(0, 0)])


def test_uniform_grid_warp_nnps_device_cache_matches_cpu_indices_cross_3d():
    rng = np.random.default_rng(2718)
    nsrc = 64
    ndst = 41
    fluid = get_particle_array(
        name='fluid',
        x=rng.random(nsrc),
        y=rng.random(nsrc),
        z=rng.random(nsrc),
        h=0.08 + 0.04 * rng.random(nsrc),
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=rng.random(ndst),
        y=rng.random(ndst),
        z=rng.random(ndst),
        h=0.08 + 0.04 * rng.random(ndst),
        backend='warp',
    )
    particles = [fluid, solid]
    cpu = LinkedListNNPS(dim=3, particles=particles, radius_scale=2.0)
    grid = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    _assert_device_cache_neighbors_match(
        cpu, grid, particles, [(0, 1), (1, 0)]
    )


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


def test_uniform_grid_warp_nnps_can_rebuild_from_device_positions():
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

    pa.gpu.x.set(np.asarray([0.0, 0.1]))
    grid.update(push=False)

    assert np.array_equal(_neighbors(grid, 0, 0, 0),
                          np.array([0, 1], dtype=np.uint32))

    grid.update()

    assert np.array_equal(_neighbors(grid, 0, 0, 0),
                          np.array([0], dtype=np.uint32))


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


# --- Multilevel GPU NNPS: level-assignment contract (kill gate, step 1) ---
#
# Range-bin, half-open convention frozen for the multilevel NNPS:
#   level k covers h in [h_ref*ratio**k, h_ref*ratio**(k+1)); level 0 is finest.
#   The top edge (h == h_ref*ratio**nlevels) is inclusive -> top level.
#   h strictly outside [h_ref, h_ref*ratio**nlevels] fails loudly.
#   Per-level support bound = radius_scale * max(assigned h) (conservative).

def test_assign_particle_levels_bins_by_half_open_ranges():
    # h_ref=0.1, ratio=2, nlevels=4 -> edges [0.1, 0.2, 0.4, 0.8, 1.6]
    #   level 0: [0.1, 0.2)  level 1: [0.2, 0.4)
    #   level 2: [0.4, 0.8)  level 3: [0.8, 1.6]
    h = np.array([0.1, 0.15, 0.2, 0.5, 0.8, 1.6], dtype=np.float64)
    levels, support = assign_particle_levels(
        h, h_ref=0.1, level_ratio=2.0, nlevels=4, radius_scale=2.0
    )
    # 0.2 and 0.8 sit on lower-closed boundaries; 1.6 is the inclusive top edge.
    assert list(levels) == [0, 0, 1, 2, 3, 3]


def test_assign_particle_levels_rejects_invalid_level_parameters():
    # Level edges h_ref*ratio**k are only strictly ascending for h_ref > 0 and
    # level_ratio > 1; otherwise binning silently inverts. Fail loudly instead.
    h = np.array([1.0])
    with pytest.raises(ValueError):  # ratio == 1 collapses all edges
        assign_particle_levels(h, h_ref=1.0, level_ratio=1.0, nlevels=3,
                               radius_scale=2.0)
    with pytest.raises(ValueError):  # ratio < 1 -> descending edges
        assign_particle_levels(h, h_ref=1.0, level_ratio=0.5, nlevels=3,
                               radius_scale=2.0)
    with pytest.raises(ValueError):  # non-positive h_ref
        assign_particle_levels(h, h_ref=0.0, level_ratio=2.0, nlevels=3,
                               radius_scale=2.0)
    with pytest.raises(ValueError):  # nlevels must be >= 1
        assign_particle_levels(h, h_ref=1.0, level_ratio=2.0, nlevels=0,
                               radius_scale=2.0)


def test_assign_particle_levels_rejects_out_of_range_h():
    # No silent clipping: h below the finest edge or above the top edge fails.
    with pytest.raises(ValueError):
        assign_particle_levels(
            np.array([0.05]), h_ref=0.1, level_ratio=2.0, nlevels=4,
            radius_scale=2.0,
        )
    with pytest.raises(ValueError):
        assign_particle_levels(
            np.array([2.0]), h_ref=0.1, level_ratio=2.0, nlevels=4,
            radius_scale=2.0,
        )


def test_assign_particle_levels_support_bound_is_conservative_max():
    # support[k] = radius_scale * max(h in level k); empty levels stay 0.
    h = np.array([0.1, 0.15, 0.5, 0.7], dtype=np.float64)
    # edges [0.1,0.2,0.4,0.8,1.6] -> levels [0,0,2,2]; levels 1 and 3 empty.
    levels, support = assign_particle_levels(
        h, h_ref=0.1, level_ratio=2.0, nlevels=4, radius_scale=2.0
    )
    assert list(levels) == [0, 0, 2, 2]
    assert np.allclose(support, [2.0 * 0.15, 0.0, 2.0 * 0.7, 0.0])


# --- Multilevel GPU NNPS: host brute-force neighbor oracle (kill gate) ---
#
# Independent pure-numpy reference for the exact symmetric pair contract
#   rij^2 < (radius_scale*h_i)^2  OR  rij^2 < (radius_scale*h_j)^2
# matching _neighbor_flags in warp_nnps (self is included: an array vs itself
# has rij=0 < support). Cross-checked against BruteForceWarpNNPS below.

def test_brute_force_neighbor_sets_matches_symmetric_cutoff_1d():
    # positions [0.0, 0.3, 1.0], h=0.2, radius_scale=2 -> support radius 0.4.
    #   dst 0: self + 0.3<0.4 -> [0,1];  dst 1: 0.3<0.4 + self -> [0,1]
    #   dst 2: 0.7 and 1.0 both > 0.4 -> [2] (self only)
    zeros = np.zeros(3)
    pa = (np.array([0.0, 0.3, 1.0]), zeros, zeros, np.full(3, 0.2))
    sets = brute_force_neighbor_sets(pa, pa, radius_scale=2.0, dim=1)
    assert [list(s) for s in sets] == [[0, 1], [0, 1], [2]]


def test_brute_force_oracle_agrees_with_brute_force_warp_nnps_2d():
    # Variable h exercises the asymmetric OR in the symmetric cutoff; the
    # host oracle must reproduce the trusted GPU BruteForceWarpNNPS exactly.
    x = [0.0, 0.2, 0.5, 0.55, 1.2]
    y = [0.0, 0.1, 0.5, 0.5, 0.0]
    h = [0.3, 0.1, 0.2, 0.05, 0.4]
    pa = get_particle_array(
        name='fluid', x=x, y=y, z=[0.0] * 5, h=h, backend='warp'
    )
    warp = BruteForceWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    warp.set_context(0, 0)
    tup = (np.array(x), np.array(y), np.zeros(5), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=2)
    for d_idx in range(5):
        actual = _neighbors(warp, 0, 0, d_idx)
        assert np.array_equal(actual, oracle[d_idx]), (d_idx, actual,
                                                       oracle[d_idx])


def test_accepted_level_pair_counts_bins_pairs_by_level():
    # 2 destinations at levels [0, 1]; 3 sources at levels [0, 0, 1].
    #   dst 0 (level 0) -> src {0(l0), 2(l1)}: (0,0)+1, (0,1)+1
    #   dst 1 (level 1) -> src {1(l0)}:        (1,0)+1
    neighbor_sets = [np.array([0, 2]), np.array([1])]
    d_levels = np.array([0, 1])
    s_levels = np.array([0, 0, 1])
    counts = accepted_level_pair_counts(
        neighbor_sets, d_levels, s_levels, nlevels=2
    )
    assert counts.tolist() == [[1, 1], [1, 0]]


# --- Multilevel GPU NNPS: MultilevelGridWarpNNPS (kill gate, step 2) ---

def test_multilevel_single_level_matches_brute_force_2d():
    # nlevels=1: the multilevel class must degenerate to exact uniform-grid /
    # brute-force behavior (cheapest kill gate). Reuses the trusted 2D config.
    x = [0.0, 0.2, 0.4, 1.5]
    y = [0.0, 0.0, 0.1, 1.5]
    h = [0.25, 0.25, 0.25, 0.25]
    pa = get_particle_array(
        name='fluid', x=x, y=y, z=[0.0] * 4, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=2.0,
        h_ref=0.25, level_ratio=2.0, nlevels=1,
    )
    bf = BruteForceWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    # Also pin against the independent numpy oracle.
    tup = (np.array(x), np.array(y), np.zeros(4), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=2)
    ml.set_context(0, 0)
    for d_idx in range(4):
        assert np.array_equal(_neighbors(ml, 0, 0, d_idx), oracle[d_idx])


def test_multilevel_four_levels_h16_cross_level_parity_3d():
    # Four discrete levels spanning h_max/h_min = 16 with genuine cross-level
    # pairs (coarse dst <-> fine src). Exact-set parity here is the core
    # correctness kill gate; the variable stencil must find neighbors whose
    # support far exceeds a fine level's cell size.
    x = [0.0, 0.8, 0.5, 1.0, 0.2, 2.0, 0.1, 5.0]
    h = [1.6, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    zeros = [0.0] * 8
    pa = get_particle_array(
        name='fluid', x=x, y=zeros, z=zeros, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=3, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=2.0, nlevels=4,
    )
    # Level assignment / support contract.
    levels, support = assign_particle_levels(
        np.array(h), h_ref=0.1, level_ratio=2.0, nlevels=4, radius_scale=2.0
    )
    assert list(levels) == [3, 3, 2, 2, 1, 1, 0, 0]
    assert np.allclose(support, [0.3, 0.6, 1.2, 3.2])

    # Exact-set parity vs the verified numpy oracle and the (uncached, robust)
    # brute-force GPU NNPS.
    tup = (np.array(x), np.array(zeros), np.array(zeros), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=3)
    bf = BruteForceWarpNNPS(dim=3, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    ml.set_context(0, 0)
    ml_sets = [_neighbors(ml, 0, 0, i) for i in range(8)]
    for i in range(8):
        assert np.array_equal(ml_sets[i], oracle[i]), (i, ml_sets[i], oracle[i])
        assert len(ml_sets[i]) == len(set(ml_sets[i].tolist())), i  # no dupes

    # The isolated fine particle (x=5.0) has only itself.
    assert list(ml_sets[7]) == [7]

    # Accepted (dst-level, src-level) matrix has real cross-level (off-diagonal)
    # mass, and matches the oracle-derived matrix.
    counts_ml = accepted_level_pair_counts(ml_sets, levels, levels, nlevels=4)
    counts_oracle = accepted_level_pair_counts(oracle, levels, levels, nlevels=4)
    assert np.array_equal(counts_ml, counts_oracle)
    off_diagonal = counts_ml.sum() - np.trace(counts_ml)
    assert off_diagonal > 0


def test_multilevel_fp32_per_level_grid_boundary_padding_1d():
    # P0 silent-omission guard: a fine particle sitting exactly on its level's
    # far edge must floor to a valid cell in [0, nx) via per-level PADDING, not
    # via the binning kernel's clamp (which would mask a padding defect).
    # h_ref=0.1, level_ratio=4, nlevels=2 -> edges [0.1, 0.4, 1.6].
    # Fine level 0 (h=0.1 -> cell_size 0.2) spans x=[0.1..0.9], extent
    # 0.8 == 4*cell_size exactly, so WITHOUT padding x=0.9 floors to cell 4==nx.
    # Spacings are deliberately off the 0.2 support so no PAIR sits on the
    # neighbor cutoff (which would make fp32 and the fp64 oracle disagree); this
    # fixture isolates grid-cell-boundary padding, not cutoff rounding. The
    # coarse particle sits at 0.5 so its 0.8 support clearly covers every fine
    # particle (max dist 0.4), again avoiding a cutoff-boundary pair.
    x = [0.1, 0.25, 0.55, 0.72, 0.9, 0.5]
    h = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4]  # last is the coarse (level 1) particle
    zeros = [0.0] * 6
    pa = get_particle_array(
        name='fluid', x=x, y=zeros, z=zeros, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=1, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=4.0, nlevels=2,
    )

    info = ml.level_grid_info(0)
    lv = info['levels']
    ox = info['origin_x']
    cs = info['cell_size']
    nxs = info['nx']
    assert list(lv) == [0, 0, 0, 0, 0, 1]
    # Pre-clamp cell index (computed in the device fp32 precision) is in range
    # for every particle -- especially the far-edge fine particle at x=0.9.
    xf = np.asarray(x, dtype=ox.dtype)
    for i in range(6):
        k = int(lv[i])
        ix = int(np.floor((xf[i] - ox[k]) / cs[k]))
        assert 0 <= ix < nxs[k], (i, ix, nxs[k])
    # Regression witness: WITHOUT padding the far fine particle would land on
    # cell nx (out of range) -- documents why the per-level origin is padded.
    unpadded_nx = int(np.ceil((0.9 - 0.1) / 0.2))
    assert int(np.floor((0.9 - 0.1) / 0.2)) == unpadded_nx  # == nx => OOB

    # Full neighbor-set parity (fp32 GPU) vs oracle and uncached brute force.
    tup = (np.array(x), np.array(zeros), np.array(zeros), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=1)
    bf = BruteForceWarpNNPS(dim=1, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    ml.set_context(0, 0)
    ml_sets = [_neighbors(ml, 0, 0, i) for i in range(6)]
    for i in range(6):
        assert np.array_equal(ml_sets[i], oracle[i]), (i, ml_sets[i], oracle[i])
    # Cross-level pair reaching the far-edge fine particle (idx4 at x=0.9) from
    # the coarse particle (idx5): its 0.8 support spans the fine AABB.
    assert 4 in ml_sets[5].tolist() and 5 in ml_sets[4].tolist()


def test_multilevel_empty_interior_levels_are_well_formed_2d():
    # Levels 1 and 3 are unpopulated: their per-level metadata must be
    # degenerate-safe (no cells, no NaN origin) and traversal must skip them,
    # while cross-level (level 0 <-> level 2) neighbors stay exact.
    x = [0.0, 0.15, 0.2, 0.8]
    h = [0.1, 0.15, 0.5, 0.7]   # edges [0.1,0.2,0.4,0.8,1.6] -> levels [0,0,2,2]
    zeros = [0.0] * 4
    pa = get_particle_array(
        name='fluid', x=x, y=zeros, z=zeros, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=2.0, nlevels=4,
    )
    levels, support = assign_particle_levels(
        np.array(h), h_ref=0.1, level_ratio=2.0, nlevels=4, radius_scale=2.0
    )
    assert list(levels) == [0, 0, 2, 2]
    assert np.allclose(support, [0.3, 0.0, 1.4, 0.0])

    # Empty levels 1 and 3 allocate no cells (nx == 0), not a degenerate grid.
    info = ml.level_grid_info(0)
    assert info['nx'][1] == 0 and info['nx'][3] == 0
    assert info['nx'][0] > 0 and info['nx'][2] > 0

    tup = (np.array(x), np.array(zeros), np.array(zeros), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=2)
    bf = BruteForceWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    ml.set_context(0, 0)
    ml_sets = [_neighbors(ml, 0, 0, i) for i in range(4)]
    for i in range(4):
        assert np.array_equal(ml_sets[i], oracle[i]), (i, ml_sets[i], oracle[i])
        assert len(ml_sets[i]) == len(set(ml_sets[i].tolist())), i

    # Accepted level-pair matrix: empty levels 1,3 have all-zero rows/cols;
    # only the (0,0),(0,2),(2,0),(2,2) blocks are populated.
    counts = accepted_level_pair_counts(ml_sets, levels, levels, nlevels=4)
    assert counts[1].sum() == 0 and counts[3].sum() == 0
    assert counts[:, 1].sum() == 0 and counts[:, 3].sum() == 0
    assert counts[0, 2] > 0 and counts[2, 0] > 0

    # Repeated update() rebuilds empty-level metadata cleanly (idempotent sets).
    ml.update()
    ml.set_context(0, 0)
    for i in range(4):
        assert np.array_equal(_neighbors(ml, 0, 0, i), oracle[i]), i


def test_multilevel_gradual_ratio_1_2_adjacent_levels_2d():
    # Four closely-spaced levels (ratio 1.2) with near-equal per-level cell
    # sizes; guards adjacent-level edge binning and the coarse-into-finer
    # query-cell-range rounding. h are strictly interior to their bins so the
    # fp32 device path bins identically to the fp64 oracle. p5 is isolated to
    # exercise exclusion, not just connectivity.
    x = [0.0, 0.1, 0.2, 0.3, 0.15, 2.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.15, 0.0]
    h = [0.11, 0.13, 0.15, 0.19, 0.19, 0.11]
    zeros = [0.0] * 6
    pa = get_particle_array(
        name='fluid', x=x, y=y, z=zeros, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=1.2, nlevels=4,
    )
    levels, support = assign_particle_levels(
        np.array(h), h_ref=0.1, level_ratio=1.2, nlevels=4, radius_scale=2.0
    )
    assert list(levels) == [0, 1, 2, 3, 3, 0]
    assert np.allclose(support, [0.22, 0.26, 0.30, 0.38])

    tup = (np.array(x), np.array(y), np.array(zeros), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=2)
    bf = BruteForceWarpNNPS(dim=2, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    ml.set_context(0, 0)
    ml_sets = [_neighbors(ml, 0, 0, i) for i in range(6)]
    for i in range(6):
        assert np.array_equal(ml_sets[i], oracle[i]), (i, ml_sets[i], oracle[i])
        assert len(ml_sets[i]) == len(set(ml_sets[i].tolist())), i
    # p5 is isolated (only itself); adjacent-level pairs are found.
    assert list(ml_sets[5]) == [5]
    counts = accepted_level_pair_counts(ml_sets, levels, levels, nlevels=4)
    for a, b in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]:
        assert counts[a, b] > 0, (a, b)


def test_multilevel_cross_array_traversal_and_ownership_2d():
    # Two independently-leveled arrays; every source/destination context must
    # match brute force, neighbor indices stay in the SOURCE array's own 0-based
    # space, and each source array owns one cached multilevel structure.
    fx, fy = [0.0, 0.2, 0.5], [0.0, 0.0, 0.0]
    fh = [0.1, 0.15, 0.4]                 # edges [0.1,0.2,0.4,0.8] -> [0,0,2]
    sx, sy = [0.1, 0.6], [0.0, 0.0]
    sh = [0.2, 0.6]                        # -> [1,2]
    fluid = get_particle_array(
        name='fluid', x=fx, y=fy, z=[0.0] * 3, h=fh, backend='warp'
    )
    solid = get_particle_array(
        name='solid', x=sx, y=sy, z=[0.0] * 2, h=sh, backend='warp'
    )
    particles = [fluid, solid]
    ml = MultilevelGridWarpNNPS(
        dim=2, particles=particles, radius_scale=2.0,
        h_ref=0.1, level_ratio=2.0, nlevels=3,
    )
    bf = BruteForceWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    # Per-array level assignment is independent.
    fl, _ = assign_particle_levels(np.array(fh), 0.1, 2.0, 3, 2.0)
    sl, _ = assign_particle_levels(np.array(sh), 0.1, 2.0, 3, 2.0)
    assert list(fl) == [0, 0, 2] and list(sl) == [1, 2]

    arrays = {0: (np.array(fx), np.array(fy), np.zeros(3), np.array(fh)),
              1: (np.array(sx), np.array(sy), np.zeros(2), np.array(sh))}
    contexts = [(0, 0), (1, 1), (0, 1), (1, 0)]
    _assert_all_neighbors_match(bf, ml, particles, contexts)
    for src_index, dst_index in contexts:
        oracle = brute_force_neighbor_sets(
            arrays[dst_index], arrays[src_index], radius_scale=2.0, dim=2
        )
        ml.set_context(src_index, dst_index)
        ndst = particles[dst_index].get_number_of_particles()
        for d_idx in range(ndst):
            got = _neighbors(ml, src_index, dst_index, d_idx)
            assert np.array_equal(got, oracle[d_idx]), (src_index, dst_index,
                                                        d_idx, got, oracle[d_idx])
            # Indices are in the source array's own 0-based space.
            nsrc = particles[src_index].get_number_of_particles()
            assert got.size == 0 or int(got.max()) < nsrc

    # Each source array owns a distinct cached multilevel structure.
    assert set(ml._ml.keys()) == {0, 1}
    assert ml._ml[0] is not ml._ml[1]


def test_multilevel_particles_at_spatial_bounds_3d():
    # Particles at the geometric min/max corners of each level's occupied
    # region (all axes) must bin to valid cells via per-level padding, and keep
    # their colocated cross-level neighbors. edges [0.1,0.2,0.4].
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
               (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    fx = [c[0] for c in corners] + [0.5]
    fy = [c[1] for c in corners] + [0.5]
    fz = [c[2] for c in corners] + [0.5]
    fh = [0.1] * 9                                  # fine, level 0
    cx, cy, cz, ch = [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.4, 0.4]  # coarse l1
    x = fx + cx
    y = fy + cy
    z = fz + cz
    h = fh + ch
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h, backend='warp')
    ml = MultilevelGridWarpNNPS(
        dim=3, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=2.0, nlevels=2,
    )
    levels, support = assign_particle_levels(
        np.array(h), 0.1, 2.0, 2, 2.0
    )
    assert list(levels) == [0] * 9 + [1, 1]
    assert np.allclose(support, [0.2, 0.8])

    # Every particle -- including the 8 corner extremes -- floors to a valid
    # in-range cell on every axis (pre-clamp), via per-level origin padding.
    info = ml.level_grid_info(0)
    ox, oy, oz = info['origin_x'], info['origin_y'], info['origin_z']
    cs, nxs, nys, nzs = (info['cell_size'], info['nx'], info['ny'], info['nz'])
    xa = np.asarray(x, dtype=ox.dtype)
    ya = np.asarray(y, dtype=ox.dtype)
    za = np.asarray(z, dtype=ox.dtype)
    for i in range(len(x)):
        k = int(levels[i])
        ix = int(np.floor((xa[i] - ox[k]) / cs[k]))
        iy = int(np.floor((ya[i] - oy[k]) / cs[k]))
        iz = int(np.floor((za[i] - oz[k]) / cs[k]))
        assert 0 <= ix < nxs[k] and 0 <= iy < nys[k] and 0 <= iz < nzs[k], (
            i, ix, iy, iz, nxs[k], nys[k], nzs[k])

    tup = (np.array(x), np.array(y), np.array(z), np.array(h))
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=3)
    bf = BruteForceWarpNNPS(dim=3, particles=[pa], radius_scale=2.0)
    _assert_all_neighbors_match(bf, ml, [pa], [(0, 0)])
    ml.set_context(0, 0)
    for i in range(len(x)):
        got = _neighbors(ml, 0, 0, i)
        assert np.array_equal(got, oracle[i]), (i, got, oracle[i])
        assert len(got) == len(set(got.tolist())), i
    # Corner colocated cross-level pairs retained: coarse idx9 at (0,0,0) <->
    # fine idx0 at (0,0,0); coarse idx10 at (1,1,1) <-> fine idx7 at (1,1,1).
    assert 0 in _neighbors(ml, 0, 0, 9).tolist()
    assert 7 in _neighbors(ml, 0, 0, 10).tolist()


def test_multilevel_no_coordinate_host_readback_on_warm_update_3d():
    # Device residency: a warm update(push=False) followed by a query must NOT
    # pull per-particle x/y/z/h back to the host. Only O(nlevels) scalar
    # metadata readback is permitted (and is not on the coordinate arrays).
    x = [0.0, 0.8, 0.5, 1.0, 0.2, 2.0, 0.1, 5.0]
    h = [1.6, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    zeros = [0.0] * 8
    pa = get_particle_array(
        name='fluid', x=x, y=zeros, z=zeros, h=h, backend='warp'
    )
    ml = MultilevelGridWarpNNPS(
        dim=3, particles=[pa], radius_scale=2.0,
        h_ref=0.1, level_ratio=2.0, nlevels=4,
    )

    reads = []

    def _spy(name, orig):
        def wrapped():
            reads.append(name)
            return orig()
        return wrapped

    patched = [n for n in ('x', 'y', 'z', 'h')]
    for name in patched:
        arr = getattr(pa.gpu, name)
        arr.get = _spy(name, arr.get)
    try:
        ml.update(push=False)          # warm rebuild
        ml.set_context(0, 0)
        for i in range(8):             # traversal / neighbor build
            _neighbors(ml, 0, 0, i)
    finally:
        for name in patched:
            arr = getattr(pa.gpu, name)
            if 'get' in arr.__dict__:
                del arr.__dict__['get']

    assert reads == [], (
        "warm update/query pulled coordinates to host: %r" % reads
    )


def _uniform_candidate_pairs(uniform, src_index, dst_index):
    # Total source particles the uniform grid's fixed 3x3x3 stencil scans,
    # summed over destinations -- the candidate work to beat.
    grid = uniform._build_grid(src_index)
    counts = grid['counts'].numpy()
    b = uniform._bounds
    cs = uniform.cell_size
    nx, ny, nz = b['nx'], b['ny'], b['nz']
    dim = uniform.dim
    dst = uniform.particles[dst_index].gpu
    dx, dy, dz = dst.x.get(), dst.y.get(), dst.z.get()

    def cell0(c, cmin, n):
        return min(max(int(np.floor((c - cmin) / cs)), 0), n - 1)

    total = 0
    for i in range(len(dx)):
        ix0 = cell0(dx[i], b['xmin'], nx)
        iy0 = cell0(dy[i], b['ymin'], ny) if dim > 1 else 0
        iz0 = cell0(dz[i], b['zmin'], nz) if dim > 2 else 0
        for dzc in (-1, 0, 1):
            for dyc in (-1, 0, 1):
                for dxc in (-1, 0, 1):
                    ix, iy, iz = ix0 + dxc, iy0 + dyc, iz0 + dzc
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        total += int(counts[ix + iy * nx + iz * nx * ny])
    return total


def test_multilevel_clustered_refinement_candidate_scaling_3d():
    # Localized refinement: a few coarse particles (h=0.8) far enough to inflate
    # the global hmax plus a dense fine cluster (h=0.05, spacing 0.08 != the
    # 0.1 support so no pair sits exactly on the cutoff). The uniform grid's
    # global cell (rs*hmax=1.6) lumps the whole fine cluster into ~one cell, so
    # every fine destination scans it entirely; the multilevel grid confines
    # fine-fine scanning to local fine cells. Accepted sets stay identical;
    # candidate work drops >=4x.
    g = np.linspace(0.0, 0.8, 11)                            # spacing 0.08
    FX, FY, FZ = np.meshgrid(g, g, g, indexing='ij')
    fx, fy, fz = FX.ravel(), FY.ravel(), FZ.ravel()          # 1331 fine
    fh = np.full(fx.size, 0.05)
    corners = np.array([(a, b_, c) for a in (0.0, 4.0)
                        for b_ in (0.0, 4.0) for c in (0.0, 4.0)])
    cx, cy, cz = corners[:, 0], corners[:, 1], corners[:, 2]  # 8 coarse
    ch = np.full(cx.size, 0.8)
    x = np.concatenate([fx, cx])
    y = np.concatenate([fy, cy])
    z = np.concatenate([fz, cz])
    h = np.concatenate([fh, ch])
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h, backend='warp')

    ml = MultilevelGridWarpNNPS(
        dim=3, particles=[pa], radius_scale=2.0,
        h_ref=0.05, level_ratio=2.0, nlevels=4,
    )
    nfine = fx.size
    levels, _ = assign_particle_levels(h, 0.05, 2.0, 4, 2.0)
    assert set(levels[:nfine].tolist()) == {0}
    assert set(levels[nfine:].tolist()) == {3}

    # Accepted-set parity: multilevel == numpy oracle == uniform grid.
    tup = (x, y, z, h)
    oracle = brute_force_neighbor_sets(tup, tup, radius_scale=2.0, dim=3)
    uniform = UniformGridWarpNNPS(dim=3, particles=[pa], radius_scale=2.0)
    ml.set_context(0, 0)
    uniform.set_context(0, 0)
    accepted = 0
    for i in range(len(x)):
        ml_i = _neighbors(ml, 0, 0, i)
        assert np.array_equal(ml_i, oracle[i]), i
        assert np.array_equal(_neighbors(uniform, 0, 0, i), oracle[i]), i
        assert len(ml_i) == len(set(ml_i.tolist())), i
        accepted += len(oracle[i])

    # Candidate work: multilevel <= 0.25 * uniform (>= 4x reduction).
    ml_cand = ml.candidate_pairs(0, 0)
    uniform_cand = _uniform_candidate_pairs(uniform, 0, 0)
    assert ml_cand >= accepted            # candidates are a superset of accepted
    assert ml_cand * 4 <= uniform_cand, (ml_cand, uniform_cand, accepted)
