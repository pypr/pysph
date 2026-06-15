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


    @wp.func
    def _cubic_dwdq_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
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
        if rij > wp.float64(1.0e-12):
            if q > wp.float64(2.0):
                val = wp.float64(0.0)
            elif q > wp.float64(1.0):
                val = -wp.float64(0.75) * tmp * tmp
            else:
                val = (
                    -wp.float64(3.0) * q *
                    (wp.float64(1.0) - wp.float64(0.75) * q)
                )
        return val * fac


    @wp.func
    def _cubic_dwdq_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
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
        if rij > wp.float32(1.0e-12):
            if q > wp.float32(2.0):
                val = wp.float32(0.0)
            elif q > wp.float32(1.0):
                val = -wp.float32(0.75) * tmp * tmp
            else:
                val = (
                    -wp.float32(3.0) * q *
                    (wp.float32(1.0) - wp.float32(0.75) * q)
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
    def _isothermal_eos_f64(
            rho: wp.array(dtype=wp.float64),
            p: wp.array(dtype=wp.float64),
            rho0: wp.float64,
            c02: wp.float64,
            p0: wp.float64,
    ):
        i = wp.tid()
        p[i] = p0 + c02 * (rho[i] - rho0)


    @wp.kernel
    def _isothermal_eos_f32(
            rho: wp.array(dtype=wp.float32),
            p: wp.array(dtype=wp.float32),
            rho0: wp.float32,
            c02: wp.float32,
            p0: wp.float32,
    ):
        i = wp.tid()
        p[i] = p0 + c02 * (rho[i] - rho0)


    @wp.kernel
    def _continuity_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            s_m: wp.array(dtype=wp.float64),
            s_u: wp.array(dtype=wp.float64),
            s_v: wp.array(dtype=wp.float64),
            s_w: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_u: wp.array(dtype=wp.float64),
            d_v: wp.array(dtype=wp.float64),
            d_w: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_arho: wp.array(dtype=wp.float64),
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
            tmp = wp.float64(0.0)
            if rij > wp.float64(1.0e-12):
                tmp = _cubic_dwdq_f64(rij, hij, dim) / (hij * rij)
            dwx = tmp * dx
            dwy = tmp * dy
            dwz = tmp * dz
            vijx = d_u[i] - s_u[j]
            vijy = d_v[i] - s_v[j]
            vijz = d_w[i] - s_w[j]
            total += s_m[j] * (vijx*dwx + vijy*dwy + vijz*dwz)
        d_arho[i] = total


    @wp.kernel
    def _continuity_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            s_m: wp.array(dtype=wp.float32),
            s_u: wp.array(dtype=wp.float32),
            s_v: wp.array(dtype=wp.float32),
            s_w: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_u: wp.array(dtype=wp.float32),
            d_v: wp.array(dtype=wp.float32),
            d_w: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_arho: wp.array(dtype=wp.float32),
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
            tmp = wp.float32(0.0)
            if rij > wp.float32(1.0e-12):
                tmp = _cubic_dwdq_f32(rij, hij, dim) / (hij * rij)
            dwx = tmp * dx
            dwy = tmp * dy
            dwz = tmp * dz
            vijx = d_u[i] - s_u[j]
            vijy = d_v[i] - s_v[j]
            vijz = d_w[i] - s_w[j]
            total += s_m[j] * (vijx*dwx + vijy*dwy + vijz*dwz)
        d_arho[i] = total


    @wp.kernel
    def _pressure_gradient_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            s_m: wp.array(dtype=wp.float64),
            s_rho: wp.array(dtype=wp.float64),
            s_p: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_rho: wp.array(dtype=wp.float64),
            d_p: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_au: wp.array(dtype=wp.float64),
            d_av: wp.array(dtype=wp.float64),
            d_aw: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        au = wp.float64(0.0)
        av = wp.float64(0.0)
        aw = wp.float64(0.0)
        rhoi21 = wp.float64(1.0) / (d_rho[i] * d_rho[i])
        tmpi = d_p[i] * rhoi21
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
            grad = wp.float64(0.0)
            if rij > wp.float64(1.0e-12):
                grad = _cubic_dwdq_f64(rij, hij, dim) / (hij * rij)
            dwx = grad * dx
            dwy = grad * dy
            dwz = grad * dz
            rhoj21 = wp.float64(1.0) / (s_rho[j] * s_rho[j])
            tmp = tmpi + s_p[j] * rhoj21
            fac = -s_m[j] * tmp
            au += fac * dwx
            av += fac * dwy
            aw += fac * dwz
        d_au[i] = au
        d_av[i] = av
        d_aw[i] = aw


    @wp.kernel
    def _pressure_gradient_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            s_m: wp.array(dtype=wp.float32),
            s_rho: wp.array(dtype=wp.float32),
            s_p: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_rho: wp.array(dtype=wp.float32),
            d_p: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            d_au: wp.array(dtype=wp.float32),
            d_av: wp.array(dtype=wp.float32),
            d_aw: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        au = wp.float32(0.0)
        av = wp.float32(0.0)
        aw = wp.float32(0.0)
        rhoi21 = wp.float32(1.0) / (d_rho[i] * d_rho[i])
        tmpi = d_p[i] * rhoi21
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
            grad = wp.float32(0.0)
            if rij > wp.float32(1.0e-12):
                grad = _cubic_dwdq_f32(rij, hij, dim) / (hij * rij)
            dwx = grad * dx
            dwy = grad * dy
            dwz = grad * dz
            rhoj21 = wp.float32(1.0) / (s_rho[j] * s_rho[j])
            tmp = tmpi + s_p[j] * rhoj21
            fac = -s_m[j] * tmp
            au += fac * dwx
            av += fac * dwy
            aw += fac * dwz
        d_au[i] = au
        d_av[i] = av
        d_aw[i] = aw


    @wp.kernel
    def _euler_step_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            w: wp.array(dtype=wp.float64),
            au: wp.array(dtype=wp.float64),
            av: wp.array(dtype=wp.float64),
            aw: wp.array(dtype=wp.float64),
            dt: wp.float64,
            dim: wp.int32,
    ):
        i = wp.tid()
        u[i] = u[i] + dt * au[i]
        v[i] = v[i] + dt * av[i]
        w[i] = w[i] + dt * aw[i]
        x[i] = x[i] + dt * u[i]
        if dim > wp.int32(1):
            y[i] = y[i] + dt * v[i]
        if dim > wp.int32(2):
            z[i] = z[i] + dt * w[i]


    @wp.kernel
    def _euler_step_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            u: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32),
            au: wp.array(dtype=wp.float32),
            av: wp.array(dtype=wp.float32),
            aw: wp.array(dtype=wp.float32),
            dt: wp.float32,
            dim: wp.int32,
    ):
        i = wp.tid()
        u[i] = u[i] + dt * au[i]
        v[i] = v[i] + dt * av[i]
        w[i] = w[i] + dt * aw[i]
        x[i] = x[i] + dt * u[i]
        if dim > wp.int32(1):
            y[i] = y[i] + dt * v[i]
        if dim > wp.int32(2):
            z[i] = z[i] + dt * w[i]


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


def _ensure_warp_helper(pa, device):
    if pa.gpu is None or getattr(pa.gpu, 'backend', None) != 'warp':
        pa.set_device_helper(WarpDeviceHelper(pa, backend='warp',
                                              device=device))


def _ensure_property(pa, prop, device):
    if prop not in pa.properties:
        pa.add_property(prop)
        if pa.gpu is not None and getattr(pa.gpu, 'backend', None) == 'warp':
            pa.gpu.add_prop(prop, pa.properties[prop])
    _ensure_warp_helper(pa, device)


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
    _ensure_property(dst_pa, out_prop, nnps.device)

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


def compute_isothermal_eos(pa, rho0, c0, p0=0.0, out_prop='p',
                           device=None, push=True):
    """Compute PySPH ``IsothermalEOS`` on a Warp ParticleArray."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_isothermal_eos")

    device = wp.get_device(device)
    _ensure_property(pa, out_prop, device)
    if push:
        pa.gpu.push('rho', out_prop)
    rho = pa.gpu.get_device_array('rho')
    out = pa.gpu.get_device_array(out_prop)
    n = pa.gpu.get_number_of_particles()
    if rho.dtype == np.float32:
        kernel = _isothermal_eos_f32
        rho0 = np.float32(rho0)
        c02 = np.float32(c0*c0)
        p0 = np.float32(p0)
    else:
        kernel = _isothermal_eos_f64
        rho0 = np.float64(rho0)
        c02 = np.float64(c0*c0)
        p0 = np.float64(p0)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[rho.dev, out.dev, rho0, c02, p0],
            device=device,
        )
        wp.synchronize_device(device)
    return out


def compute_continuity(nnps, src_index=0, dst_index=0, out_prop='arho'):
    """Compute PySPH ``ContinuityEquation`` with Warp."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_continuity")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_property(dst_pa, out_prop, nnps.device)

    src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'u', 'v', 'w')
    dst_pa.gpu.push('x', 'y', 'z', 'h', 'u', 'v', 'w', out_prop)
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    out = dst.get_device_array(out_prop)
    ndst = dst.get_number_of_particles()
    if src.x.dtype == np.float32:
        kernel = _continuity_f32
    else:
        kernel = _continuity_f64

    if ndst > 0:
        wp.launch(
            kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.u.dev, src.v.dev, src.w.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.u.dev, dst.v.dev, dst.w.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), out.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return out


def compute_pressure_gradient(nnps, src_index=0, dst_index=0,
                              out_props=('au', 'av', 'aw'), push=True):
    """Compute the inviscid pressure-gradient part of WCSPH momentum."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_pressure_gradient")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    for prop in out_props:
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'p')
        dst_pa.gpu.push('x', 'y', 'z', 'h', 'rho', 'p', *out_props)
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    au = dst.get_device_array(out_props[0])
    av = dst.get_device_array(out_props[1])
    aw = dst.get_device_array(out_props[2])
    ndst = dst.get_number_of_particles()
    if src.x.dtype == np.float32:
        kernel = _pressure_gradient_f32
    else:
        kernel = _pressure_gradient_f64

    if ndst > 0:
        wp.launch(
            kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.rho.dev, src.p.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.rho.dev, dst.p.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim),
                au.dev, av.dev, aw.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return au, av, aw


def euler_step(pa, dt, dim=3, device=None, push=True):
    """Advance position and velocity using already-computed acceleration."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for euler_step")

    device = wp.get_device(device)
    _ensure_warp_helper(pa, device)
    if push:
        pa.gpu.push('x', 'y', 'z', 'u', 'v', 'w', 'au', 'av', 'aw')
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if gpu.x.dtype == np.float32:
        kernel = _euler_step_f32
        dt = np.float32(dt)
    else:
        kernel = _euler_step_f64
        dt = np.float64(dt)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.x.dev, gpu.y.dev, gpu.z.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev,
                gpu.au.dev, gpu.av.dev, gpu.aw.dev,
                dt, np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.x, gpu.y, gpu.z, gpu.u, gpu.v, gpu.w


def wc_sph_euler_step(nnps, pa_index=0, dt=1.0e-4, rho0=1000.0,
                      c0=20.0, p0=0.0):
    """Run one minimal WCSPH-style device step.

    The step computes summation density, isothermal pressure, inviscid pressure
    acceleration, and a simple Euler velocity/position update on the device.
    """
    pa = nnps.particles[pa_index]
    compute_summation_density(nnps, pa_index, pa_index)
    compute_isothermal_eos(
        pa, rho0=rho0, c0=c0, p0=p0, device=nnps.device, push=False
    )
    compute_pressure_gradient(nnps, pa_index, pa_index, push=False)
    return euler_step(pa, dt=dt, dim=nnps.dim, device=nnps.device,
                      push=False)
