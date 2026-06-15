"""Small Warp SPH equation kernels used by the GPU migration prototype."""

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from pysph.base.warp_device_helper import WarpDeviceHelper


if wp is not None:
    @wp.func
    def _cubic_spline_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
        h1 = wp.float64(1.0) / h
        q = rij * h1
        fac = wp.float64(2.0) / wp.float64(3.0)
        if dim == wp.int32(2):
            fac = wp.float64(10.0) / (
                wp.float64(7.0) * wp.float64(3.141592653589793)
            )
        elif dim == wp.int32(3):
            fac = wp.float64(1.0) / wp.float64(3.141592653589793)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float64(0.0)
        tmp = wp.float64(2.0) - q
        if q > wp.float64(2.0):
            val = wp.float64(0.0)
        elif q > wp.float64(1.0):
            val = wp.float64(0.25) * tmp * tmp * tmp
        else:
            val = (
                wp.float64(1.0) -
                wp.float64(1.5) * q * q *
                (wp.float64(1.0) - wp.float64(0.5) * q)
            )
        return val * fac


    @wp.func
    def _cubic_spline_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
        h1 = wp.float32(1.0) / h
        q = rij * h1
        fac = wp.float32(2.0) / wp.float32(3.0)
        if dim == wp.int32(2):
            fac = wp.float32(10.0) / (
                wp.float32(7.0) * wp.float32(3.141592653589793)
            )
        elif dim == wp.int32(3):
            fac = wp.float32(1.0) / wp.float32(3.141592653589793)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float32(0.0)
        tmp = wp.float32(2.0) - q
        if q > wp.float32(2.0):
            val = wp.float32(0.0)
        elif q > wp.float32(1.0):
            val = wp.float32(0.25) * tmp * tmp * tmp
        else:
            val = (
                wp.float32(1.0) -
                wp.float32(1.5) * q * q *
                (wp.float32(1.0) - wp.float32(0.5) * q)
            )
        return val * fac


    @wp.kernel
    def _summation_density_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            s_m: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_rho: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        total = wp.float64(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            dx = d_x[i] - s_x[j]
            dy = wp.float64(0.0)
            dz = wp.float64(0.0)
            if dim > wp.int32(1):
                dy = d_y[i] - s_y[j]
            if dim > wp.int32(2):
                dz = d_z[i] - s_z[j]
            rij = wp.sqrt(dx*dx + dy*dy + dz*dz)
            hij = wp.float64(0.5) * (d_h[i] + s_h[j])
            total += s_m[j] * _cubic_spline_f64(rij, hij, dim)
        d_rho[i] = total


    @wp.kernel
    def _summation_density_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            s_m: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_rho: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        total = wp.float32(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            dx = d_x[i] - s_x[j]
            dy = wp.float32(0.0)
            dz = wp.float32(0.0)
            if dim > wp.int32(1):
                dy = d_y[i] - s_y[j]
            if dim > wp.int32(2):
                dz = d_z[i] - s_z[j]
            rij = wp.sqrt(dx*dx + dy*dy + dz*dz)
            hij = wp.float32(0.5) * (d_h[i] + s_h[j])
            total += s_m[j] * _cubic_spline_f32(rij, hij, dim)
        d_rho[i] = total


def compute_summation_density(nnps, src_index=0, dst_index=0,
                              out_prop='rho'):
    """Compute standard SPH summation density with Warp.

    This mirrors ``pysph.sph.basic_equations.SummationDensity`` for one
    source/destination pair using PySPH's standard ``HIJ`` convention:
    ``HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])``.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_summation_density")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    if out_prop not in dst_pa.properties:
        dst_pa.add_property(out_prop)
        if dst_pa.gpu is None or getattr(dst_pa.gpu, 'backend', None) != 'warp':
            dst_pa.set_device_helper(
                WarpDeviceHelper(dst_pa, backend='warp', device=nnps.device)
            )
        else:
            dst_pa.gpu.add_prop(out_prop, dst_pa.properties[out_prop])

    src_pa.gpu.push('x', 'y', 'z', 'h', 'm')
    dst_pa.gpu.push('x', 'y', 'z', 'h', out_prop)
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    out = dst.get_device_array(out_prop)
    ndst = dst.get_number_of_particles()

    if src.x.dtype == np.float32:
        kernel = _summation_density_f32
    else:
        kernel = _summation_density_f64

    if ndst > 0:
        wp.launch(
            kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), out.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return out
