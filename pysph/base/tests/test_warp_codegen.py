"""Focused tests for the dynamic Warp equation-group code generator (ADR-0003).

These exercise the generator mechanism in isolation: that a kernel templated
from equation-block snippets compiles, runs, and is cached, that a generated
kernel resolves and calls the existing device ``wp.func`` helpers, and that the
unioned signature/metadata is correct. Physics parity of the real SPH blocks
against the hand-written kernels lives in ``test_warp_sph.py``.
"""

import numpy as np
import pytest

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

pytest.importorskip('warp')

import warp as wp

from pysph.base.kernels import CubicSpline
from pysph.base.warp_codegen import (
    WarpEquation, build_group_kernel, clear_kernel_cache,
)
import pysph.base.warp_sph as ws


def _device():
    return wp.get_device()


def _arr(values, dtype):
    return wp.array(np.asarray(values, dtype=dtype), device=_device())


class _SumMass(WarpEquation):
    """Toy block: accumulate neighbor mass (no geometry, no wp.func calls)."""
    src_arrays = ('m',)
    out_arrays = ('total',)

    def loop(self):
        return "        _acc_total += s_m[j]"


class _KernelSum(WarpEquation):
    """Toy block: accumulate the SPH kernel value over neighbors.

    Exercises shared-geometry emission and a generated call into the existing
    device ``_kernel_value_*`` ``wp.func``.
    """
    out_arrays = ('wsum',)
    requires = ('rij', 'hij', 'wij')

    def loop(self):
        return "        _acc_wsum += wij"


def _launch_manual(group, arrays, dim, kernel_id, scalars=None):
    """Bind device arrays/scalars in the generator's canonical order."""
    scalars = scalars or {}
    inputs = [arrays['s_' + n] for n in group.src_names]
    inputs += [arrays['d_' + n] for n in group.dst_names]
    inputs += [arrays['starts'], arrays['lengths'], arrays['neighbors']]
    inputs += [np.int32(dim), np.int32(kernel_id)]
    inputs += [group.dtype(scalars[n]) for n in group.scalar_names]
    inputs += [arrays['d_' + n] for n in group.out_names]
    n = arrays['_n']
    wp.launch(group.kernel, dim=n, inputs=inputs, device=_device())
    wp.synchronize_device(_device())


def _launch_grid_manual(group, arrays, dim, kernel_id, scalars=None):
    """Bind device arrays/scalars for a ``grid`` kernel in canonical order."""
    scalars = scalars or {}
    inputs = [arrays['s_' + n] for n in group.src_names]
    inputs += [arrays['d_' + n] for n in group.dst_names]
    inputs += [
        arrays['cell_starts'], arrays['cell_counts'], arrays['cell_particles'],
        group.dtype(arrays['xmin']), group.dtype(arrays['ymin']),
        group.dtype(arrays['zmin']), group.dtype(arrays['cell_size']),
        np.int32(arrays['nx']), np.int32(arrays['ny']), np.int32(arrays['nz']),
        np.int32(arrays['ncells']), group.dtype(arrays['radius_scale']),
    ]
    inputs += [np.int32(dim), np.int32(kernel_id)]
    inputs += [group.dtype(scalars[n]) for n in group.scalar_names]
    inputs += [arrays['d_' + n] for n in group.out_names]
    wp.launch(group.kernel, dim=arrays['_n'], inputs=inputs, device=_device())
    wp.synchronize_device(_device())


def test_generated_group_kernel_compiles_runs_and_is_cached():
    clear_kernel_cache()
    group = build_group_kernel([_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS)
    assert group.src_names == ['m']
    assert group.out_names == ['total']
    assert group.scalar_names == []

    # 3 particles, each neighboring all three.
    arrays = {
        's_m': _arr([1.0, 2.0, 3.0], np.float32),
        'starts': _arr([0, 3, 6], np.int32),
        'lengths': _arr([3, 3, 3], np.int32),
        'neighbors': _arr([0, 1, 2, 0, 1, 2, 0, 1, 2], np.uint32),
        'd_total': wp.zeros(3, dtype=wp.float32, device=_device()),
        '_n': 3,
    }
    _launch_manual(group, arrays, dim=2, kernel_id=0)
    total = arrays['d_total'].numpy()
    assert np.allclose(total, [6.0, 6.0, 6.0])

    # Building the same group again returns the identical cached object.
    group2 = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS
    )
    assert group2 is group


def test_generated_kernel_calls_existing_device_wp_func():
    clear_kernel_cache()
    group = build_group_kernel(
        [_KernelSum()], np.float32, ws._WARP_DEVICE_FUNCS
    )
    # Geometry auto-adds x, y, z, h to both source and destination signature.
    assert group.src_names == ['x', 'y', 'z', 'h']
    assert group.dst_names == ['x', 'y', 'z', 'h']

    r = 0.3
    h = 0.35
    arrays = {
        's_x': _arr([0.0, r], np.float32), 's_y': _arr([0.0, 0.0], np.float32),
        's_z': _arr([0.0, 0.0], np.float32),
        's_h': _arr([h, h], np.float32),
        'd_x': _arr([0.0, r], np.float32), 'd_y': _arr([0.0, 0.0], np.float32),
        'd_z': _arr([0.0, 0.0], np.float32),
        'd_h': _arr([h, h], np.float32),
        'starts': _arr([0, 1], np.int32),
        'lengths': _arr([1, 1], np.int32),
        'neighbors': _arr([1, 0], np.uint32),
        'd_wsum': wp.zeros(2, dtype=wp.float32, device=_device()),
        '_n': 2,
    }
    _launch_manual(group, arrays, dim=2, kernel_id=0)
    wsum = arrays['d_wsum'].numpy()

    cpu = CubicSpline(dim=2)
    expected = cpu.kernel([r, 0.0, 0.0], r, 0.5 * (h + h))
    assert np.allclose(wsum, [expected, expected], rtol=1e-5, atol=1e-6)


def test_generator_unions_signature_and_scalars_in_order():
    group = build_group_kernel(
        ws._WCSPH_CONTINUITY_BLOCKS, np.float64, ws._WARP_DEVICE_FUNCS
    )
    # Distinct outputs from all four blocks, in block order.
    assert group.out_names == ['au', 'av', 'aw', 'arho', 'ax', 'ay', 'az']
    # Scalars collected across blocks in block order, de-duplicated.
    assert group.scalar_names == ['alpha', 'beta', 'eps']
    # Velocity + geometry arrays are auto-added and shared (no duplicates).
    for name in ('x', 'y', 'z', 'h', 'u', 'v', 'w', 'm', 'rho', 'p', 'cs'):
        assert name in group.src_names
    assert len(group.src_names) == len(set(group.src_names))


def test_grid_neighbor_mode_caches_distinctly_and_forces_geometry():
    # ADR-0004: grid mode is part of the structural cache key (distinct from
    # flat) and forces positions + h into the signature even for a block that
    # declares no geometry, because the support cutoff needs them.
    clear_kernel_cache()
    flat = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS, neighbor_mode='flat'
    )
    grid = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS, neighbor_mode='grid'
    )
    assert flat.neighbor_mode == 'flat'
    assert grid.neighbor_mode == 'grid'
    assert grid is not flat
    for n in ('x', 'y', 'z', 'h'):
        assert n in grid.src_names and n in grid.dst_names
    assert 'x' not in flat.src_names  # flat keeps the minimal signature
    grid2 = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS, neighbor_mode='grid'
    )
    assert grid2 is grid


def test_grid_mode_kernel_sum_matches_reference_single_cell():
    # A grid kernel over a single cell holding both particles must visit each
    # particle's self-pair and its in-cutoff neighbor, calling the device
    # kernel-value wp.func through the post-cutoff geometry.
    clear_kernel_cache()
    group = build_group_kernel(
        [_KernelSum()], np.float32, ws._WARP_DEVICE_FUNCS, neighbor_mode='grid'
    )
    r = 0.3
    h = 0.35
    arrays = {
        's_x': _arr([0.0, r], np.float32), 's_y': _arr([0.0, 0.0], np.float32),
        's_z': _arr([0.0, 0.0], np.float32), 's_h': _arr([h, h], np.float32),
        'd_x': _arr([0.0, r], np.float32), 'd_y': _arr([0.0, 0.0], np.float32),
        'd_z': _arr([0.0, 0.0], np.float32), 'd_h': _arr([h, h], np.float32),
        'cell_starts': _arr([0], np.int32),
        'cell_counts': _arr([2], np.int32),
        'cell_particles': _arr([0, 1], np.uint32),
        'xmin': -1.0, 'ymin': -1.0, 'zmin': -1.0, 'cell_size': 10.0,
        'nx': 1, 'ny': 1, 'nz': 1, 'ncells': 1, 'radius_scale': 2.0,
        'd_wsum': wp.zeros(2, dtype=wp.float32, device=_device()),
        '_n': 2,
    }
    _launch_grid_manual(group, arrays, dim=2, kernel_id=0)
    wsum = arrays['d_wsum'].numpy()

    cpu = CubicSpline(dim=2)
    w_self = cpu.kernel([0.0, 0.0, 0.0], 0.0, h)
    w_pair = cpu.kernel([r, 0.0, 0.0], r, h)
    expected = w_self + w_pair
    assert np.allclose(wsum, [expected, expected], rtol=1e-5, atol=1e-6)


def test_accumulate_outputs_adds_to_existing_output():
    # accumulate_outputs=True seeds _acc from the existing d_<out>[i] so the
    # group adds to (read-modify-writes) the destination arrays instead of
    # overwriting; it is a distinct cached kernel from the overwrite variant.
    clear_kernel_cache()
    overwrite = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS,
        accumulate_outputs=False,
    )
    accumulate = build_group_kernel(
        [_SumMass()], np.float32, ws._WARP_DEVICE_FUNCS,
        accumulate_outputs=True,
    )
    assert accumulate is not overwrite

    # 3 particles, each neighboring all three; sum of masses is 6. Seed the
    # output with 10 so accumulate yields 16 (overwrite would yield 6).
    arrays = {
        's_m': _arr([1.0, 2.0, 3.0], np.float32),
        'starts': _arr([0, 3, 6], np.int32),
        'lengths': _arr([3, 3, 3], np.int32),
        'neighbors': _arr([0, 1, 2, 0, 1, 2, 0, 1, 2], np.uint32),
        'd_total': _arr([10.0, 10.0, 10.0], np.float32),
        '_n': 3,
    }
    _launch_manual(accumulate, arrays, dim=2, kernel_id=0)
    assert np.allclose(arrays['d_total'].numpy(), [16.0, 16.0, 16.0])


def test_periodic_grid_kernel_compiles_and_is_distinct():
    # The periodic (minimum-image) variant is a distinct cached kernel from the
    # non-periodic one and only valid in grid mode.
    clear_kernel_cache()
    nonper = build_group_kernel(
        [_KernelSum()], np.float32, ws._WARP_DEVICE_FUNCS,
        neighbor_mode='grid', periodic=False,
    )
    per = build_group_kernel(
        [_KernelSum()], np.float32, ws._WARP_DEVICE_FUNCS,
        neighbor_mode='grid', periodic=True,
    )
    assert per is not nonper
    assert 'box_lx' in per.source and 'box_lx' not in nonper.source
    with pytest.raises(ValueError):
        build_group_kernel(
            [_KernelSum()], np.float32, ws._WARP_DEVICE_FUNCS,
            neighbor_mode='flat', periodic=True,
        )


def test_unknown_shared_quantity_is_rejected():
    class _Bad(WarpEquation):
        out_arrays = ('q',)
        requires = ('bogus',)

        def loop(self):
            return "        _acc_q += TYPE(1.0)"

    with pytest.raises(ValueError):
        build_group_kernel([_Bad()], np.float32, ws._WARP_DEVICE_FUNCS)
