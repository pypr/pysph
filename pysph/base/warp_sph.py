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


    @wp.func
    def _gaussian_spline_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
        h1 = wp.float64(1.0) / h
        q = rij * h1
        fac = wp.float64(0.5641895835477563)
        if dim == wp.int32(2):
            fac = fac * wp.float64(0.5641895835477563)
        elif dim == wp.int32(3):
            fac = fac * wp.float64(0.5641895835477563) * wp.float64(0.5641895835477563)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float64(0.0)
        if q < wp.float64(3.0):
            val = wp.exp(-q*q)
        return val * fac


    @wp.func
    def _gaussian_spline_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
        h1 = wp.float32(1.0) / h
        q = rij * h1
        fac = wp.float32(0.5641895835477563)
        if dim == wp.int32(2):
            fac = fac * wp.float32(0.5641895835477563)
        elif dim == wp.int32(3):
            fac = fac * wp.float32(0.5641895835477563) * wp.float32(0.5641895835477563)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float32(0.0)
        if q < wp.float32(3.0):
            val = wp.exp(-q*q)
        return val * fac


    @wp.func
    def _gaussian_dwdq_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
        h1 = wp.float64(1.0) / h
        q = rij * h1
        fac = wp.float64(0.5641895835477563)
        if dim == wp.int32(2):
            fac = fac * wp.float64(0.5641895835477563)
        elif dim == wp.int32(3):
            fac = fac * wp.float64(0.5641895835477563) * wp.float64(0.5641895835477563)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float64(0.0)
        if rij > wp.float64(1.0e-12) and q < wp.float64(3.0):
            val = -wp.float64(2.0) * q * wp.exp(-q*q)
        return val * fac


    @wp.func
    def _gaussian_dwdq_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
        h1 = wp.float32(1.0) / h
        q = rij * h1
        fac = wp.float32(0.5641895835477563)
        if dim == wp.int32(2):
            fac = fac * wp.float32(0.5641895835477563)
        elif dim == wp.int32(3):
            fac = fac * wp.float32(0.5641895835477563) * wp.float32(0.5641895835477563)

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float32(0.0)
        if rij > wp.float32(1.0e-12) and q < wp.float32(3.0):
            val = -wp.float32(2.0) * q * wp.exp(-q*q)
        return val * fac


    @wp.func
    def _kernel_value_f64(
            rij: wp.float64, h: wp.float64, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_spline_f64(rij, h, dim)
        return _cubic_spline_f64(rij, h, dim)


    @wp.func
    def _kernel_value_f32(
            rij: wp.float32, h: wp.float32, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_spline_f32(rij, h, dim)
        return _cubic_spline_f32(rij, h, dim)


    @wp.func
    def _kernel_dwdq_f64(
            rij: wp.float64, h: wp.float64, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_dwdq_f64(rij, h, dim)
        return _cubic_dwdq_f64(rij, h, dim)


    @wp.func
    def _kernel_dwdq_f32(
            rij: wp.float32, h: wp.float32, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_dwdq_f32(rij, h, dim)
        return _cubic_dwdq_f32(rij, h, dim)


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
            kernel_id: wp.int32,
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
            total += s_m[j] * _kernel_value_f64(rij, hij, dim, kernel_id)
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
    def _tait_eos_f64(
            rho: wp.array(dtype=wp.float64),
            p: wp.array(dtype=wp.float64),
            cs: wp.array(dtype=wp.float64),
            rho0: wp.float64,
            rho01: wp.float64,
            c0: wp.float64,
            gamma: wp.float64,
            gamma1: wp.float64,
            b: wp.float64,
            p0: wp.float64,
    ):
        i = wp.tid()
        ratio = rho[i] * rho01
        tmp = wp.pow(ratio, gamma)
        p[i] = p0 + b * (tmp - wp.float64(1.0))
        cs[i] = c0 * wp.pow(ratio, gamma1)


    @wp.kernel
    def _tait_eos_f32(
            rho: wp.array(dtype=wp.float32),
            p: wp.array(dtype=wp.float32),
            cs: wp.array(dtype=wp.float32),
            rho0: wp.float32,
            rho01: wp.float32,
            c0: wp.float32,
            gamma: wp.float32,
            gamma1: wp.float32,
            b: wp.float32,
            p0: wp.float32,
    ):
        i = wp.tid()
        ratio = rho[i] * rho01
        tmp = wp.pow(ratio, gamma)
        p[i] = p0 + b * (tmp - wp.float32(1.0))
        cs[i] = c0 * wp.pow(ratio, gamma1)


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
            kernel_id: wp.int32,
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
                tmp = _kernel_dwdq_f64(rij, hij, dim, kernel_id) / (hij * rij)
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
            kernel_id: wp.int32,
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
                tmp = _kernel_dwdq_f32(rij, hij, dim, kernel_id) / (hij * rij)
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
            kernel_id: wp.int32,
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
                grad = _kernel_dwdq_f64(rij, hij, dim, kernel_id) / (hij * rij)
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
            kernel_id: wp.int32,
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
                grad = _kernel_dwdq_f32(rij, hij, dim, kernel_id) / (hij * rij)
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
    def _artificial_viscosity_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            s_m: wp.array(dtype=wp.float64),
            s_rho: wp.array(dtype=wp.float64),
            s_cs: wp.array(dtype=wp.float64),
            s_u: wp.array(dtype=wp.float64),
            s_v: wp.array(dtype=wp.float64),
            s_w: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_rho: wp.array(dtype=wp.float64),
            d_cs: wp.array(dtype=wp.float64),
            d_u: wp.array(dtype=wp.float64),
            d_v: wp.array(dtype=wp.float64),
            d_w: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            kernel_id: wp.int32,
            alpha: wp.float64,
            beta: wp.float64,
            d_au: wp.array(dtype=wp.float64),
            d_av: wp.array(dtype=wp.float64),
            d_aw: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        au = d_au[i]
        av = d_av[i]
        aw = d_aw[i]
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
            vijx = d_u[i] - s_u[j]
            vijy = wp.float64(0.0)
            vijz = wp.float64(0.0)
            if dim > wp.int32(1):
                vijy = d_v[i] - s_v[j]
            if dim > wp.int32(2):
                vijz = d_w[i] - s_w[j]
            vdotx = vijx*dx + vijy*dy + vijz*dz
            if vdotx < wp.float64(0.0):
                rij2 = dx*dx + dy*dy + dz*dz
                rij = wp.sqrt(rij2)
                hij = wp.float64(0.5) * (d_h[i] + s_h[j])
                grad = wp.float64(0.0)
                if rij > wp.float64(1.0e-12):
                    grad = _kernel_dwdq_f64(rij, hij, dim, kernel_id) / (hij * rij)
                mu = hij * vdotx / (rij2 + wp.float64(0.01)*hij*hij)
                rhoij1 = wp.float64(2.0) / (d_rho[i] + s_rho[j])
                cij = wp.float64(0.5) * (d_cs[i] + s_cs[j])
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                fac = -s_m[j] * piij
                au += fac * grad * dx
                av += fac * grad * dy
                aw += fac * grad * dz
        d_au[i] = au
        d_av[i] = av
        d_aw[i] = aw


    @wp.kernel
    def _artificial_viscosity_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            s_m: wp.array(dtype=wp.float32),
            s_rho: wp.array(dtype=wp.float32),
            s_cs: wp.array(dtype=wp.float32),
            s_u: wp.array(dtype=wp.float32),
            s_v: wp.array(dtype=wp.float32),
            s_w: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_rho: wp.array(dtype=wp.float32),
            d_cs: wp.array(dtype=wp.float32),
            d_u: wp.array(dtype=wp.float32),
            d_v: wp.array(dtype=wp.float32),
            d_w: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            kernel_id: wp.int32,
            alpha: wp.float32,
            beta: wp.float32,
            d_au: wp.array(dtype=wp.float32),
            d_av: wp.array(dtype=wp.float32),
            d_aw: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        au = d_au[i]
        av = d_av[i]
        aw = d_aw[i]
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
            vijx = d_u[i] - s_u[j]
            vijy = wp.float32(0.0)
            vijz = wp.float32(0.0)
            if dim > wp.int32(1):
                vijy = d_v[i] - s_v[j]
            if dim > wp.int32(2):
                vijz = d_w[i] - s_w[j]
            vdotx = vijx*dx + vijy*dy + vijz*dz
            if vdotx < wp.float32(0.0):
                rij2 = dx*dx + dy*dy + dz*dz
                rij = wp.sqrt(rij2)
                hij = wp.float32(0.5) * (d_h[i] + s_h[j])
                grad = wp.float32(0.0)
                if rij > wp.float32(1.0e-12):
                    grad = _kernel_dwdq_f32(rij, hij, dim, kernel_id) / (hij * rij)
                mu = hij * vdotx / (rij2 + wp.float32(0.01)*hij*hij)
                rhoij1 = wp.float32(2.0) / (d_rho[i] + s_rho[j])
                cij = wp.float32(0.5) * (d_cs[i] + s_cs[j])
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                fac = -s_m[j] * piij
                au += fac * grad * dx
                av += fac * grad * dy
                aw += fac * grad * dz
        d_au[i] = au
        d_av[i] = av
        d_aw[i] = aw


    @wp.kernel
    def _xsph_correction_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            s_m: wp.array(dtype=wp.float64),
            s_rho: wp.array(dtype=wp.float64),
            s_u: wp.array(dtype=wp.float64),
            s_v: wp.array(dtype=wp.float64),
            s_w: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_rho: wp.array(dtype=wp.float64),
            d_u: wp.array(dtype=wp.float64),
            d_v: wp.array(dtype=wp.float64),
            d_w: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            kernel_id: wp.int32,
            eps: wp.float64,
            d_ax: wp.array(dtype=wp.float64),
            d_ay: wp.array(dtype=wp.float64),
            d_az: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        ax = wp.float64(0.0)
        ay = wp.float64(0.0)
        az = wp.float64(0.0)
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
            wij = _kernel_value_f64(rij, hij, dim, kernel_id)
            rhoij1 = wp.float64(2.0) / (d_rho[i] + s_rho[j])
            tmp = -eps * s_m[j] * wij * rhoij1
            ax += tmp * (d_u[i] - s_u[j])
            if dim > wp.int32(1):
                ay += tmp * (d_v[i] - s_v[j])
            if dim > wp.int32(2):
                az += tmp * (d_w[i] - s_w[j])
        d_ax[i] = ax
        d_ay[i] = ay
        d_az[i] = az


    @wp.kernel
    def _xsph_correction_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            s_m: wp.array(dtype=wp.float32),
            s_rho: wp.array(dtype=wp.float32),
            s_u: wp.array(dtype=wp.float32),
            s_v: wp.array(dtype=wp.float32),
            s_w: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_rho: wp.array(dtype=wp.float32),
            d_u: wp.array(dtype=wp.float32),
            d_v: wp.array(dtype=wp.float32),
            d_w: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            kernel_id: wp.int32,
            eps: wp.float32,
            d_ax: wp.array(dtype=wp.float32),
            d_ay: wp.array(dtype=wp.float32),
            d_az: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        ax = wp.float32(0.0)
        ay = wp.float32(0.0)
        az = wp.float32(0.0)
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
            wij = _kernel_value_f32(rij, hij, dim, kernel_id)
            rhoij1 = wp.float32(2.0) / (d_rho[i] + s_rho[j])
            tmp = -eps * s_m[j] * wij * rhoij1
            ax += tmp * (d_u[i] - s_u[j])
            if dim > wp.int32(1):
                ay += tmp * (d_v[i] - s_v[j])
            if dim > wp.int32(2):
                az += tmp * (d_w[i] - s_w[j])
        d_ax[i] = ax
        d_ay[i] = ay
        d_az[i] = az


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
    def _leapfrog_kick_f64(
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
        if dim > wp.int32(1):
            v[i] = v[i] + dt * av[i]
        if dim > wp.int32(2):
            w[i] = w[i] + dt * aw[i]


    @wp.kernel
    def _leapfrog_kick_f32(
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
        if dim > wp.int32(1):
            v[i] = v[i] + dt * av[i]
        if dim > wp.int32(2):
            w[i] = w[i] + dt * aw[i]


    @wp.kernel
    def _leapfrog_drift_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            w: wp.array(dtype=wp.float64),
            dt: wp.float64,
            dim: wp.int32,
    ):
        i = wp.tid()
        x[i] = x[i] + dt * u[i]
        if dim > wp.int32(1):
            y[i] = y[i] + dt * v[i]
        if dim > wp.int32(2):
            z[i] = z[i] + dt * w[i]


    @wp.kernel
    def _leapfrog_drift_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            u: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32),
            dt: wp.float32,
            dim: wp.int32,
    ):
        i = wp.tid()
        x[i] = x[i] + dt * u[i]
        if dim > wp.int32(1):
            y[i] = y[i] + dt * v[i]
        if dim > wp.int32(2):
            z[i] = z[i] + dt * w[i]


    @wp.kernel
    def _leapfrog_drift_xsph_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            w: wp.array(dtype=wp.float64),
            ax: wp.array(dtype=wp.float64),
            ay: wp.array(dtype=wp.float64),
            az: wp.array(dtype=wp.float64),
            dt: wp.float64,
            dim: wp.int32,
    ):
        i = wp.tid()
        x[i] = x[i] + dt * (u[i] + ax[i])
        if dim > wp.int32(1):
            y[i] = y[i] + dt * (v[i] + ay[i])
        if dim > wp.int32(2):
            z[i] = z[i] + dt * (w[i] + az[i])


    @wp.kernel
    def _leapfrog_drift_xsph_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            u: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32),
            ax: wp.array(dtype=wp.float32),
            ay: wp.array(dtype=wp.float32),
            az: wp.array(dtype=wp.float32),
            dt: wp.float32,
            dim: wp.int32,
    ):
        i = wp.tid()
        x[i] = x[i] + dt * (u[i] + ax[i])
        if dim > wp.int32(1):
            y[i] = y[i] + dt * (v[i] + ay[i])
        if dim > wp.int32(2):
            z[i] = z[i] + dt * (w[i] + az[i])


    @wp.kernel
    def _wcsph_save_state_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            w: wp.array(dtype=wp.float64),
            rho: wp.array(dtype=wp.float64),
            x0: wp.array(dtype=wp.float64),
            y0: wp.array(dtype=wp.float64),
            z0: wp.array(dtype=wp.float64),
            u0: wp.array(dtype=wp.float64),
            v0: wp.array(dtype=wp.float64),
            w0: wp.array(dtype=wp.float64),
            rho0: wp.array(dtype=wp.float64),
            dim: wp.int32,
    ):
        i = wp.tid()
        x0[i] = x[i]
        u0[i] = u[i]
        rho0[i] = rho[i]
        if dim > wp.int32(1):
            y0[i] = y[i]
            v0[i] = v[i]
        if dim > wp.int32(2):
            z0[i] = z[i]
            w0[i] = w[i]


    @wp.kernel
    def _wcsph_save_state_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            u: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32),
            rho: wp.array(dtype=wp.float32),
            x0: wp.array(dtype=wp.float32),
            y0: wp.array(dtype=wp.float32),
            z0: wp.array(dtype=wp.float32),
            u0: wp.array(dtype=wp.float32),
            v0: wp.array(dtype=wp.float32),
            w0: wp.array(dtype=wp.float32),
            rho0: wp.array(dtype=wp.float32),
            dim: wp.int32,
    ):
        i = wp.tid()
        x0[i] = x[i]
        u0[i] = u[i]
        rho0[i] = rho[i]
        if dim > wp.int32(1):
            y0[i] = y[i]
            v0[i] = v[i]
        if dim > wp.int32(2):
            z0[i] = z[i]
            w0[i] = w[i]


    @wp.kernel
    def _wcsph_pec_stage_f64(
            x0: wp.array(dtype=wp.float64),
            y0: wp.array(dtype=wp.float64),
            z0: wp.array(dtype=wp.float64),
            u0: wp.array(dtype=wp.float64),
            v0: wp.array(dtype=wp.float64),
            w0: wp.array(dtype=wp.float64),
            rho0: wp.array(dtype=wp.float64),
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            w: wp.array(dtype=wp.float64),
            rho: wp.array(dtype=wp.float64),
            au: wp.array(dtype=wp.float64),
            av: wp.array(dtype=wp.float64),
            aw: wp.array(dtype=wp.float64),
            ax: wp.array(dtype=wp.float64),
            ay: wp.array(dtype=wp.float64),
            az: wp.array(dtype=wp.float64),
            arho: wp.array(dtype=wp.float64),
            dt_factor: wp.float64,
            dim: wp.int32,
            use_xsph: wp.int32,
    ):
        i = wp.tid()
        adv_x = u[i]
        adv_y = v[i]
        adv_z = w[i]
        if use_xsph:
            adv_x = adv_x + ax[i]
            adv_y = adv_y + ay[i]
            adv_z = adv_z + az[i]
        u[i] = u0[i] + dt_factor * au[i]
        rho[i] = rho0[i] + dt_factor * arho[i]
        x[i] = x0[i] + dt_factor * adv_x
        if dim > wp.int32(1):
            v[i] = v0[i] + dt_factor * av[i]
            y[i] = y0[i] + dt_factor * adv_y
        if dim > wp.int32(2):
            w[i] = w0[i] + dt_factor * aw[i]
            z[i] = z0[i] + dt_factor * adv_z


    @wp.kernel
    def _wcsph_pec_stage_f32(
            x0: wp.array(dtype=wp.float32),
            y0: wp.array(dtype=wp.float32),
            z0: wp.array(dtype=wp.float32),
            u0: wp.array(dtype=wp.float32),
            v0: wp.array(dtype=wp.float32),
            w0: wp.array(dtype=wp.float32),
            rho0: wp.array(dtype=wp.float32),
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            u: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32),
            rho: wp.array(dtype=wp.float32),
            au: wp.array(dtype=wp.float32),
            av: wp.array(dtype=wp.float32),
            aw: wp.array(dtype=wp.float32),
            ax: wp.array(dtype=wp.float32),
            ay: wp.array(dtype=wp.float32),
            az: wp.array(dtype=wp.float32),
            arho: wp.array(dtype=wp.float32),
            dt_factor: wp.float32,
            dim: wp.int32,
            use_xsph: wp.int32,
    ):
        i = wp.tid()
        adv_x = u[i]
        adv_y = v[i]
        adv_z = w[i]
        if use_xsph:
            adv_x = adv_x + ax[i]
            adv_y = adv_y + ay[i]
            adv_z = adv_z + az[i]
        u[i] = u0[i] + dt_factor * au[i]
        rho[i] = rho0[i] + dt_factor * arho[i]
        x[i] = x0[i] + dt_factor * adv_x
        if dim > wp.int32(1):
            v[i] = v0[i] + dt_factor * av[i]
            y[i] = y0[i] + dt_factor * adv_y
        if dim > wp.int32(2):
            w[i] = w0[i] + dt_factor * aw[i]
            z[i] = z0[i] + dt_factor * adv_z


    @wp.kernel
    def _wcsph_dt_factors_f64(
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_u: wp.array(dtype=wp.float64),
            d_v: wp.array(dtype=wp.float64),
            d_w: wp.array(dtype=wp.float64),
            d_au: wp.array(dtype=wp.float64),
            d_av: wp.array(dtype=wp.float64),
            d_aw: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            c0: wp.float64,
            d_dt_cfl: wp.array(dtype=wp.float64),
            d_dt_force: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        cfl_fac = wp.float64(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            dx = d_x[i] - d_x[j]
            dy = wp.float64(0.0)
            dz = wp.float64(0.0)
            vijx = d_u[i] - d_u[j]
            vijy = wp.float64(0.0)
            vijz = wp.float64(0.0)
            if dim > wp.int32(1):
                dy = d_y[i] - d_y[j]
                vijy = d_v[i] - d_v[j]
            if dim > wp.int32(2):
                dz = d_z[i] - d_z[j]
                vijz = d_w[i] - d_w[j]
            rij2 = dx*dx + dy*dy + dz*dz
            if rij2 > wp.float64(1.0e-12):
                hij = wp.float64(0.5) * (d_h[i] + d_h[j])
                vdotx = vijx*dx + vijy*dy + vijz*dz
                factor = wp.abs(hij * vdotx / rij2) + c0
                cfl_fac = wp.max(cfl_fac, factor)
        d_dt_cfl[i] = cfl_fac
        d_dt_force[i] = d_au[i]*d_au[i] + d_av[i]*d_av[i] + d_aw[i]*d_aw[i]


    @wp.kernel
    def _wcsph_dt_factors_f32(
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_u: wp.array(dtype=wp.float32),
            d_v: wp.array(dtype=wp.float32),
            d_w: wp.array(dtype=wp.float32),
            d_au: wp.array(dtype=wp.float32),
            d_av: wp.array(dtype=wp.float32),
            d_aw: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            dim: wp.int32,
            c0: wp.float32,
            d_dt_cfl: wp.array(dtype=wp.float32),
            d_dt_force: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        cfl_fac = wp.float32(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            dx = d_x[i] - d_x[j]
            dy = wp.float32(0.0)
            dz = wp.float32(0.0)
            vijx = d_u[i] - d_u[j]
            vijy = wp.float32(0.0)
            vijz = wp.float32(0.0)
            if dim > wp.int32(1):
                dy = d_y[i] - d_y[j]
                vijy = d_v[i] - d_v[j]
            if dim > wp.int32(2):
                dz = d_z[i] - d_z[j]
                vijz = d_w[i] - d_w[j]
            rij2 = dx*dx + dy*dy + dz*dz
            if rij2 > wp.float32(1.0e-12):
                hij = wp.float32(0.5) * (d_h[i] + d_h[j])
                vdotx = vijx*dx + vijy*dy + vijz*dz
                factor = wp.abs(hij * vdotx / rij2) + c0
                cfl_fac = wp.max(cfl_fac, factor)
        d_dt_cfl[i] = cfl_fac
        d_dt_force[i] = d_au[i]*d_au[i] + d_av[i]*d_av[i] + d_aw[i]*d_aw[i]


    @wp.kernel
    def _wcsph_dt_init_f64(
            max_cfl: wp.array(dtype=wp.float64),
            max_force: wp.array(dtype=wp.float64),
            min_h: wp.array(dtype=wp.float64),
            out_dt: wp.array(dtype=wp.float64),
    ):
        max_cfl[0] = wp.float64(0.0)
        max_force[0] = wp.float64(0.0)
        min_h[0] = wp.float64(1.0e30)
        out_dt[0] = wp.float64(0.0)


    @wp.kernel
    def _wcsph_dt_init_f32(
            max_cfl: wp.array(dtype=wp.float32),
            max_force: wp.array(dtype=wp.float32),
            min_h: wp.array(dtype=wp.float32),
            out_dt: wp.array(dtype=wp.float32),
    ):
        max_cfl[0] = wp.float32(0.0)
        max_force[0] = wp.float32(0.0)
        min_h[0] = wp.float32(1.0e30)
        out_dt[0] = wp.float32(0.0)


    @wp.kernel
    def _wcsph_dt_reduce_f64(
            h: wp.array(dtype=wp.float64),
            dt_cfl: wp.array(dtype=wp.float64),
            dt_force: wp.array(dtype=wp.float64),
            max_cfl: wp.array(dtype=wp.float64),
            max_force: wp.array(dtype=wp.float64),
            min_h: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        wp.atomic_max(max_cfl, 0, dt_cfl[i])
        wp.atomic_max(max_force, 0, dt_force[i])
        wp.atomic_min(min_h, 0, h[i])


    @wp.kernel
    def _wcsph_dt_reduce_f32(
            h: wp.array(dtype=wp.float32),
            dt_cfl: wp.array(dtype=wp.float32),
            dt_force: wp.array(dtype=wp.float32),
            max_cfl: wp.array(dtype=wp.float32),
            max_force: wp.array(dtype=wp.float32),
            min_h: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        wp.atomic_max(max_cfl, 0, dt_cfl[i])
        wp.atomic_max(max_force, 0, dt_force[i])
        wp.atomic_min(min_h, 0, h[i])


    @wp.kernel
    def _wcsph_dt_finalize_f64(
            max_cfl: wp.array(dtype=wp.float64),
            max_force: wp.array(dtype=wp.float64),
            min_h: wp.array(dtype=wp.float64),
            cfl: wp.float64,
            dt_min: wp.float64,
            dt_max: wp.float64,
            out_dt: wp.array(dtype=wp.float64),
    ):
        dt = dt_max
        if max_cfl[0] > wp.float64(0.0):
            dt = wp.min(dt, cfl * min_h[0] / max_cfl[0])
        if max_force[0] > wp.float64(0.0):
            dt_force = wp.sqrt(min_h[0] / wp.sqrt(max_force[0]))
            dt = wp.min(dt, cfl * dt_force)
        dt = wp.max(dt, dt_min)
        dt = wp.min(dt, dt_max)
        out_dt[0] = dt


    @wp.kernel
    def _wcsph_dt_finalize_f32(
            max_cfl: wp.array(dtype=wp.float32),
            max_force: wp.array(dtype=wp.float32),
            min_h: wp.array(dtype=wp.float32),
            cfl: wp.float32,
            dt_min: wp.float32,
            dt_max: wp.float32,
            out_dt: wp.array(dtype=wp.float32),
    ):
        dt = dt_max
        if max_cfl[0] > wp.float32(0.0):
            dt = wp.min(dt, cfl * min_h[0] / max_cfl[0])
        if max_force[0] > wp.float32(0.0):
            dt_force = wp.sqrt(min_h[0] / wp.sqrt(max_force[0]))
            dt = wp.min(dt, cfl * dt_force)
        dt = wp.max(dt, dt_min)
        dt = wp.min(dt, dt_max)
        out_dt[0] = dt


    @wp.func
    def _wrap_value_f64(value: wp.float64, lower: wp.float64,
                        upper: wp.float64):
        length = upper - lower
        result = value
        if length > wp.float64(0.0):
            offset = result - lower
            result = lower + offset - wp.floor(offset / length) * length
        return result


    @wp.func
    def _wrap_value_f32(value: wp.float32, lower: wp.float32,
                        upper: wp.float32):
        length = upper - lower
        result = value
        if length > wp.float32(0.0):
            offset = result - lower
            result = lower + offset - wp.floor(offset / length) * length
        return result


    @wp.kernel
    def _wrap_periodic_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            xmin: wp.float64,
            xmax: wp.float64,
            ymin: wp.float64,
            ymax: wp.float64,
            zmin: wp.float64,
            zmax: wp.float64,
            periodic_x: wp.int32,
            periodic_y: wp.int32,
            periodic_z: wp.int32,
            dim: wp.int32,
    ):
        i = wp.tid()
        if periodic_x:
            x[i] = _wrap_value_f64(x[i], xmin, xmax)
        if dim > wp.int32(1) and periodic_y:
            y[i] = _wrap_value_f64(y[i], ymin, ymax)
        if dim > wp.int32(2) and periodic_z:
            z[i] = _wrap_value_f64(z[i], zmin, zmax)


    @wp.kernel
    def _wrap_periodic_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            xmin: wp.float32,
            xmax: wp.float32,
            ymin: wp.float32,
            ymax: wp.float32,
            zmin: wp.float32,
            zmax: wp.float32,
            periodic_x: wp.int32,
            periodic_y: wp.int32,
            periodic_z: wp.int32,
            dim: wp.int32,
    ):
        i = wp.tid()
        if periodic_x:
            x[i] = _wrap_value_f32(x[i], xmin, xmax)
        if dim > wp.int32(1) and periodic_y:
            y[i] = _wrap_value_f32(y[i], ymin, ymax)
        if dim > wp.int32(2) and periodic_z:
            z[i] = _wrap_value_f32(z[i], zmin, zmax)


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
            kernel_id: wp.int32,
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
            total += s_m[j] * _kernel_value_f32(rij, hij, dim, kernel_id)
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


def _ensure_sound_speed(pa, c0, device):
    if 'cs' not in pa.properties:
        n = pa.get_number_of_particles()
        pa.add_property('cs', data=np.ones(n) * c0)
        if pa.gpu is not None and getattr(pa.gpu, 'backend', None) == 'warp':
            pa.gpu.add_prop('cs', pa.properties['cs'])
    _ensure_warp_helper(pa, device)


def _kernel_id(kernel):
    if isinstance(kernel, (int, np.integer)):
        if int(kernel) in (0, 1):
            return np.int32(kernel)
        raise ValueError("kernel id must be 0 (cubic) or 1 (gaussian)")
    name = str(kernel).lower().replace('-', '_')
    if name in ('cubic', 'cubic_spline', 'cubicspline'):
        return np.int32(0)
    if name == 'gaussian':
        return np.int32(1)
    raise ValueError("kernel must be 'cubic' or 'gaussian'")


def compute_summation_density(nnps, src_index=0, dst_index=0,
                              out_prop='rho', push=True, kernel='cubic'):
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

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm')
        dst_pa.gpu.push('x', 'y', 'z', 'h', out_prop)
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    out = dst.get_device_array(out_prop)
    ndst = dst.get_number_of_particles()
    kernel_id = _kernel_id(kernel)

    if src.x.dtype == np.float32:
        equation_kernel = _summation_density_f32
    else:
        equation_kernel = _summation_density_f64

    if ndst > 0:
        wp.launch(
            equation_kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), kernel_id, out.dev
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


def compute_tait_eos(pa, rho0, c0, gamma=7.0, p0=0.0, out_prop='p',
                     cs_prop='cs', device=None, push=True):
    """Compute PySPH ``TaitEOS`` pressure and sound speed with Warp."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_tait_eos")

    device = wp.get_device(device)
    _ensure_property(pa, out_prop, device)
    _ensure_property(pa, cs_prop, device)
    if push:
        pa.gpu.push('rho', out_prop, cs_prop)
    rho = pa.gpu.get_device_array('rho')
    out = pa.gpu.get_device_array(out_prop)
    cs = pa.gpu.get_device_array(cs_prop)
    n = pa.gpu.get_number_of_particles()
    if rho.dtype == np.float32:
        kernel = _tait_eos_f32
        rho0 = np.float32(rho0)
        rho01 = np.float32(1.0 / rho0)
        c0 = np.float32(c0)
        gamma = np.float32(gamma)
        gamma1 = np.float32(0.5 * (gamma - np.float32(1.0)))
        b = np.float32(rho0*c0*c0/gamma)
        p0 = np.float32(p0)
    else:
        kernel = _tait_eos_f64
        rho0 = np.float64(rho0)
        rho01 = np.float64(1.0 / rho0)
        c0 = np.float64(c0)
        gamma = np.float64(gamma)
        gamma1 = np.float64(0.5 * (gamma - np.float64(1.0)))
        b = np.float64(rho0*c0*c0/gamma)
        p0 = np.float64(p0)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                rho.dev, out.dev, cs.dev, rho0, rho01, c0, gamma,
                gamma1, b, p0
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return out, cs


def compute_continuity(nnps, src_index=0, dst_index=0, out_prop='arho',
                       push=True, kernel='cubic'):
    """Compute PySPH ``ContinuityEquation`` with Warp."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_continuity")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_property(dst_pa, out_prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'u', 'v', 'w')
        dst_pa.gpu.push('x', 'y', 'z', 'h', 'u', 'v', 'w', out_prop)
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    out = dst.get_device_array(out_prop)
    ndst = dst.get_number_of_particles()
    kernel_id = _kernel_id(kernel)
    if src.x.dtype == np.float32:
        equation_kernel = _continuity_f32
    else:
        equation_kernel = _continuity_f64

    if ndst > 0:
        wp.launch(
            equation_kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.u.dev, src.v.dev, src.w.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.u.dev, dst.v.dev, dst.w.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), kernel_id, out.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return out


def compute_pressure_gradient(nnps, src_index=0, dst_index=0,
                              out_props=('au', 'av', 'aw'), push=True,
                              kernel='cubic'):
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
    kernel_id = _kernel_id(kernel)
    if src.x.dtype == np.float32:
        equation_kernel = _pressure_gradient_f32
    else:
        equation_kernel = _pressure_gradient_f64

    if ndst > 0:
        wp.launch(
            equation_kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.rho.dev, src.p.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.rho.dev, dst.p.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), kernel_id,
                au.dev, av.dev, aw.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return au, av, aw


def compute_artificial_viscosity(nnps, src_index=0, dst_index=0, alpha=0.1,
                                 beta=0.0, c0=20.0,
                                 out_props=('au', 'av', 'aw'), push=True,
                                 kernel='cubic'):
    """Add Monaghan artificial viscosity to WCSPH acceleration arrays."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_artificial_viscosity")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_sound_speed(src_pa, c0, nnps.device)
    if dst_pa is not src_pa:
        _ensure_sound_speed(dst_pa, c0, nnps.device)
    for prop in out_props:
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'cs', 'u', 'v', 'w')
        dst_pa.gpu.push(
            'x', 'y', 'z', 'h', 'rho', 'cs', 'u', 'v', 'w', *out_props
        )
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    au = dst.get_device_array(out_props[0])
    av = dst.get_device_array(out_props[1])
    aw = dst.get_device_array(out_props[2])
    ndst = dst.get_number_of_particles()
    kernel_id = _kernel_id(kernel)
    if src.x.dtype == np.float32:
        equation_kernel = _artificial_viscosity_f32
        alpha = np.float32(alpha)
        beta = np.float32(beta)
    else:
        equation_kernel = _artificial_viscosity_f64
        alpha = np.float64(alpha)
        beta = np.float64(beta)

    if ndst > 0:
        wp.launch(
            equation_kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.rho.dev, src.cs.dev, src.u.dev, src.v.dev, src.w.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.rho.dev, dst.cs.dev, dst.u.dev, dst.v.dev, dst.w.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), kernel_id,
                alpha, beta, au.dev, av.dev, aw.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return au, av, aw


def compute_xsph_correction(nnps, src_index=0, dst_index=0, eps=0.5,
                            out_props=('ax', 'ay', 'az'), push=True,
                            kernel='cubic'):
    """Compute PySPH leapfrog XSPH position correction on the device."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_xsph_correction")

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    for prop in out_props:
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'u', 'v', 'w')
        dst_pa.gpu.push(
            'x', 'y', 'z', 'h', 'rho', 'u', 'v', 'w', *out_props
        )
    cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
    src = src_pa.gpu
    dst = dst_pa.gpu
    ax = dst.get_device_array(out_props[0])
    ay = dst.get_device_array(out_props[1])
    az = dst.get_device_array(out_props[2])
    ndst = dst.get_number_of_particles()
    kernel_id = _kernel_id(kernel)
    if src.x.dtype == np.float32:
        equation_kernel = _xsph_correction_f32
        eps = np.float32(eps)
    else:
        equation_kernel = _xsph_correction_f64
        eps = np.float64(eps)

    if ndst > 0:
        wp.launch(
            equation_kernel,
            dim=ndst,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev, src.m.dev,
                src.rho.dev, src.u.dev, src.v.dev, src.w.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                dst.rho.dev, dst.u.dev, dst.v.dev, dst.w.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), kernel_id, eps,
                ax.dev, ay.dev, az.dev
            ],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
    return ax, ay, az


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


def leapfrog_kick(pa, dt, dim=3, device=None, push=True):
    """Kick velocity with the current acceleration."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for leapfrog_kick")

    device = wp.get_device(device)
    _ensure_warp_helper(pa, device)
    if push:
        pa.gpu.push('u', 'v', 'w', 'au', 'av', 'aw')
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if gpu.u.dtype == np.float32:
        kernel = _leapfrog_kick_f32
        dt = np.float32(dt)
    else:
        kernel = _leapfrog_kick_f64
        dt = np.float64(dt)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.u.dev, gpu.v.dev, gpu.w.dev,
                gpu.au.dev, gpu.av.dev, gpu.aw.dev,
                dt, np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.u, gpu.v, gpu.w


def leapfrog_drift(pa, dt, dim=3, device=None, push=True):
    """Drift position with the current velocity."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for leapfrog_drift")

    device = wp.get_device(device)
    _ensure_warp_helper(pa, device)
    if push:
        pa.gpu.push('x', 'y', 'z', 'u', 'v', 'w')
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if gpu.x.dtype == np.float32:
        kernel = _leapfrog_drift_f32
        dt = np.float32(dt)
    else:
        kernel = _leapfrog_drift_f64
        dt = np.float64(dt)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.x.dev, gpu.y.dev, gpu.z.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev,
                dt, np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.x, gpu.y, gpu.z


def leapfrog_drift_xsph(pa, dt, dim=3, device=None, push=True):
    """Drift position with velocity plus precomputed XSPH correction."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for leapfrog_drift_xsph")

    device = wp.get_device(device)
    _ensure_property(pa, 'ax', device)
    _ensure_property(pa, 'ay', device)
    _ensure_property(pa, 'az', device)
    if push:
        pa.gpu.push('x', 'y', 'z', 'u', 'v', 'w', 'ax', 'ay', 'az')
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if gpu.x.dtype == np.float32:
        kernel = _leapfrog_drift_xsph_f32
        dt = np.float32(dt)
    else:
        kernel = _leapfrog_drift_xsph_f64
        dt = np.float64(dt)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.x.dev, gpu.y.dev, gpu.z.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev,
                gpu.ax.dev, gpu.ay.dev, gpu.az.dev,
                dt, np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.x, gpu.y, gpu.z


def save_wcsph_state(pa, dim=3, device=None, push=True):
    """Save WCSPH PEC reference position, velocity, and density on device."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for save_wcsph_state")

    device = wp.get_device(device)
    for prop in ('x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'):
        _ensure_property(pa, prop, device)
    if push:
        pa.gpu.push(
            'x', 'y', 'z', 'u', 'v', 'w', 'rho',
            'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'
        )
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if gpu.x.dtype == np.float32:
        kernel = _wcsph_save_state_f32
    else:
        kernel = _wcsph_save_state_f64
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.x.dev, gpu.y.dev, gpu.z.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev, gpu.rho.dev,
                gpu.x0.dev, gpu.y0.dev, gpu.z0.dev,
                gpu.u0.dev, gpu.v0.dev, gpu.w0.dev, gpu.rho0.dev,
                np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.x0, gpu.u0, gpu.rho0


def wcsph_pec_stage(pa, dt, stage=1.0, dim=3, xsph=False, device=None,
                    push=True):
    """Apply one PySPH ``WCSPHStep``-style PEC stage on the device."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for wcsph_pec_stage")

    device = wp.get_device(device)
    for prop in (
            'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0',
            'au', 'av', 'aw', 'ax', 'ay', 'az', 'arho',
    ):
        _ensure_property(pa, prop, device)
    if push:
        pa.gpu.push(
            'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0',
            'x', 'y', 'z', 'u', 'v', 'w', 'rho',
            'au', 'av', 'aw', 'ax', 'ay', 'az', 'arho'
        )
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    use_xsph = np.int32(bool(xsph))
    if gpu.x.dtype == np.float32:
        kernel = _wcsph_pec_stage_f32
        dt_factor = np.float32(dt * stage)
    else:
        kernel = _wcsph_pec_stage_f64
        dt_factor = np.float64(dt * stage)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.x0.dev, gpu.y0.dev, gpu.z0.dev,
                gpu.u0.dev, gpu.v0.dev, gpu.w0.dev, gpu.rho0.dev,
                gpu.x.dev, gpu.y.dev, gpu.z.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev, gpu.rho.dev,
                gpu.au.dev, gpu.av.dev, gpu.aw.dev,
                gpu.ax.dev, gpu.ay.dev, gpu.az.dev, gpu.arho.dev,
                dt_factor, np.int32(dim), use_xsph
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.x, gpu.y, gpu.z, gpu.u, gpu.v, gpu.w, gpu.rho


def compute_wcsph_adaptive_timestep(nnps, pa_index=0, c0=20.0, cfl=0.25,
                                    dt_min=0.0, dt_max=np.inf, push=True):
    """Compute WCSPH adaptive timestep with device reductions.

    Only the final scalar timestep is copied back to the host. Per-particle
    ``dt_cfl`` and ``dt_force`` remain on the device unless explicitly pulled.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_wcsph_adaptive_timestep")

    pa = nnps.particles[pa_index]
    _ensure_property(pa, 'dt_cfl', nnps.device)
    _ensure_property(pa, 'dt_force', nnps.device)
    if push:
        pa.gpu.push(
            'x', 'y', 'z', 'h', 'u', 'v', 'w', 'au', 'av', 'aw',
            'dt_cfl', 'dt_force'
        )
    cache = nnps.build_neighbor_cache_gpu(pa_index, pa_index)
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    dt_cfl = gpu.get_device_array('dt_cfl')
    dt_force = gpu.get_device_array('dt_force')
    if gpu.x.dtype == np.float32:
        dtype = wp.float32
        factors_kernel = _wcsph_dt_factors_f32
        init_kernel = _wcsph_dt_init_f32
        reduce_kernel = _wcsph_dt_reduce_f32
        finalize_kernel = _wcsph_dt_finalize_f32
        c0 = np.float32(c0)
        cfl = np.float32(cfl)
        dt_min = np.float32(dt_min)
        dt_max = np.float32(dt_max)
    else:
        dtype = wp.float64
        factors_kernel = _wcsph_dt_factors_f64
        init_kernel = _wcsph_dt_init_f64
        reduce_kernel = _wcsph_dt_reduce_f64
        finalize_kernel = _wcsph_dt_finalize_f64
        c0 = np.float64(c0)
        cfl = np.float64(cfl)
        dt_min = np.float64(dt_min)
        dt_max = np.float64(dt_max)

    max_cfl = wp.zeros(1, dtype=dtype, device=nnps.device)
    max_force = wp.zeros(1, dtype=dtype, device=nnps.device)
    min_h = wp.zeros(1, dtype=dtype, device=nnps.device)
    out_dt = wp.zeros(1, dtype=dtype, device=nnps.device)
    if n > 0:
        wp.launch(
            factors_kernel,
            dim=n,
            inputs=[
                gpu.x.dev, gpu.y.dev, gpu.z.dev, gpu.h.dev,
                gpu.u.dev, gpu.v.dev, gpu.w.dev,
                gpu.au.dev, gpu.av.dev, gpu.aw.dev,
                cache['starts_dev'], cache['lengths_dev'],
                cache['neighbors_dev'], np.int32(nnps.dim), c0,
                dt_cfl.dev, dt_force.dev
            ],
            device=nnps.device,
        )
        wp.launch(
            init_kernel,
            dim=1,
            inputs=[max_cfl, max_force, min_h, out_dt],
            device=nnps.device,
        )
        wp.launch(
            reduce_kernel,
            dim=n,
            inputs=[
                gpu.h.dev, dt_cfl.dev, dt_force.dev,
                max_cfl, max_force, min_h
            ],
            device=nnps.device,
        )
        wp.launch(
            finalize_kernel,
            dim=1,
            inputs=[max_cfl, max_force, min_h, cfl, dt_min, dt_max, out_dt],
            device=nnps.device,
        )
        wp.synchronize_device(nnps.device)
        return float(out_dt.numpy()[0])
    return float(dt_max)


def _periodic_bounds(bounds, dim):
    if bounds is None:
        return None
    if not isinstance(bounds, dict):
        raise TypeError("periodic bounds must be a dict or None")

    xmin = bounds.get('xmin', 0.0)
    xmax = bounds.get('xmax', xmin)
    ymin = bounds.get('ymin', 0.0)
    ymax = bounds.get('ymax', ymin)
    zmin = bounds.get('zmin', 0.0)
    zmax = bounds.get('zmax', zmin)
    periodic_x = bool(bounds.get('periodic_in_x', 'xmin' in bounds and
                                 'xmax' in bounds))
    periodic_y = bool(bounds.get('periodic_in_y', 'ymin' in bounds and
                                 'ymax' in bounds and dim > 1))
    periodic_z = bool(bounds.get('periodic_in_z', 'zmin' in bounds and
                                 'zmax' in bounds and dim > 2))
    return (
        xmin, xmax, ymin, ymax, zmin, zmax,
        periodic_x, periodic_y, periodic_z
    )


def wrap_periodic(pa, bounds, dim=3, device=None):
    """Wrap particle coordinates into a periodic box on the device."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for wrap_periodic")

    parsed = _periodic_bounds(bounds, dim)
    if parsed is None:
        return None

    device = wp.get_device(device)
    _ensure_warp_helper(pa, device)
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    if n == 0:
        return gpu.x, gpu.y, gpu.z

    xmin, xmax, ymin, ymax, zmin, zmax, px, py, pz = parsed
    if gpu.x.dtype == np.float32:
        kernel = _wrap_periodic_f32
        scalars = [
            np.float32(xmin), np.float32(xmax),
            np.float32(ymin), np.float32(ymax),
            np.float32(zmin), np.float32(zmax),
        ]
    else:
        kernel = _wrap_periodic_f64
        scalars = [
            np.float64(xmin), np.float64(xmax),
            np.float64(ymin), np.float64(ymax),
            np.float64(zmin), np.float64(zmax),
        ]

    wp.launch(
        kernel,
        dim=n,
        inputs=[
            gpu.x.dev, gpu.y.dev, gpu.z.dev,
            scalars[0], scalars[1], scalars[2], scalars[3],
            scalars[4], scalars[5],
            np.int32(px), np.int32(py), np.int32(pz), np.int32(dim)
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return gpu.x, gpu.y, gpu.z


def _compute_wcsph_acceleration(nnps, pa_index, rho0, c0, p0, alpha, beta,
                                push, eos, gamma, kernel,
                                density_mode='summation'):
    pa = nnps.particles[pa_index]
    if density_mode == 'summation':
        compute_summation_density(
            nnps, pa_index, pa_index, push=push, kernel=kernel
        )
    elif density_mode == 'continuity':
        _ensure_property(pa, 'arho', nnps.device)
        if push:
            pa.gpu.push('rho', 'arho')
    else:
        raise ValueError("density_mode must be 'summation' or 'continuity'")
    if eos == 'isothermal':
        compute_isothermal_eos(
            pa, rho0=rho0, c0=c0, p0=p0, device=nnps.device, push=False
        )
        _ensure_sound_speed(pa, c0, nnps.device)
    elif eos == 'tait':
        compute_tait_eos(
            pa, rho0=rho0, c0=c0, gamma=gamma, p0=p0,
            device=nnps.device, push=False
        )
    else:
        raise ValueError("EOS must be 'isothermal' or 'tait'")
    result = compute_pressure_gradient(
        nnps, pa_index, pa_index, push=False, kernel=kernel
    )
    if alpha != 0.0 or beta != 0.0:
        result = compute_artificial_viscosity(
            nnps, pa_index, pa_index, alpha=alpha, beta=beta, c0=c0,
            push=False, kernel=kernel
        )
    if density_mode == 'continuity':
        compute_continuity(
            nnps, pa_index, pa_index, push=False, kernel=kernel
        )
    return result


def _compute_wcsph_xsph(nnps, pa_index, xsph_eps, kernel):
    if xsph_eps is None or xsph_eps == 0.0:
        return False
    compute_xsph_correction(
        nnps, pa_index, pa_index, eps=xsph_eps, push=False,
        kernel=kernel
    )
    return True


def _wc_sph_pec_continuity_step(nnps, pa_index, dt, rho0, c0, p0,
                                periodic_bounds, push, alpha, beta, eos,
                                gamma, kernel, xsph_eps, adaptive_dt, cfl,
                                dt_min, dt_max):
    pa = nnps.particles[pa_index]
    if push:
        nnps.update(push=True)
    save_wcsph_state(pa, dim=nnps.dim, device=nnps.device, push=push)
    _compute_wcsph_acceleration(
        nnps, pa_index, rho0, c0, p0, alpha, beta, push=False,
        eos=eos, gamma=gamma, kernel=kernel, density_mode='continuity'
    )
    if adaptive_dt:
        dt = compute_wcsph_adaptive_timestep(
            nnps, pa_index=pa_index, c0=c0, cfl=cfl, dt_min=dt_min,
            dt_max=dt_max, push=False
        )
    use_xsph = _compute_wcsph_xsph(nnps, pa_index, xsph_eps, kernel)
    wcsph_pec_stage(
        pa, dt=dt, stage=0.5, dim=nnps.dim, xsph=use_xsph,
        device=nnps.device, push=False
    )
    wrap_periodic(pa, periodic_bounds, dim=nnps.dim, device=nnps.device)
    nnps.update(push=False)
    _compute_wcsph_acceleration(
        nnps, pa_index, rho0, c0, p0, alpha, beta, push=False,
        eos=eos, gamma=gamma, kernel=kernel, density_mode='continuity'
    )
    use_xsph = _compute_wcsph_xsph(nnps, pa_index, xsph_eps, kernel)
    result = wcsph_pec_stage(
        pa, dt=dt, stage=1.0, dim=nnps.dim, xsph=use_xsph,
        device=nnps.device, push=False
    )
    wrap_periodic(pa, periodic_bounds, dim=nnps.dim, device=nnps.device)
    nnps.update(push=False)
    return result, dt


def wc_sph_leapfrog_step(nnps, pa_index=0, dt=1.0e-4, rho0=1000.0,
                         c0=20.0, p0=0.0, periodic_bounds=None,
                         push=False, alpha=0.0, beta=0.0,
                         eos='isothermal', gamma=7.0, kernel='cubic',
                         xsph_eps=None, adaptive_dt=False, cfl=0.25,
                         dt_min=0.0, dt_max=np.inf, return_dt=False,
                         density_mode='summation'):
    """Run one minimal WCSPH KDK leapfrog step on the device.

    ``push`` defaults to ``False`` so repeated calls keep the Warp arrays as the
    source of truth. Pass ``push=True`` only when host ParticleArray values were
    intentionally changed before the step.
    """
    pa = nnps.particles[pa_index]
    if density_mode == 'continuity':
        result, dt = _wc_sph_pec_continuity_step(
            nnps, pa_index, dt, rho0, c0, p0, periodic_bounds, push,
            alpha, beta, eos, gamma, kernel, xsph_eps, adaptive_dt, cfl,
            dt_min, dt_max
        )
        if return_dt:
            return result, dt
        return result
    if density_mode != 'summation':
        raise ValueError("density_mode must be 'summation' or 'continuity'")
    if push:
        nnps.update(push=True)
    _compute_wcsph_acceleration(
        nnps, pa_index, rho0, c0, p0, alpha, beta, push=push,
        eos=eos, gamma=gamma, kernel=kernel, density_mode='summation'
    )
    if adaptive_dt:
        dt = compute_wcsph_adaptive_timestep(
            nnps, pa_index=pa_index, c0=c0, cfl=cfl, dt_min=dt_min,
            dt_max=dt_max, push=False
        )
    leapfrog_kick(pa, dt=0.5*dt, dim=nnps.dim, device=nnps.device,
                  push=False)
    if xsph_eps is None or xsph_eps == 0.0:
        leapfrog_drift(
            pa, dt=dt, dim=nnps.dim, device=nnps.device, push=False
        )
    else:
        compute_xsph_correction(
            nnps, pa_index, pa_index, eps=xsph_eps, push=False,
            kernel=kernel
        )
        leapfrog_drift_xsph(
            pa, dt=dt, dim=nnps.dim, device=nnps.device, push=False
        )
    wrap_periodic(pa, periodic_bounds, dim=nnps.dim, device=nnps.device)
    nnps.update(push=False)
    _compute_wcsph_acceleration(
        nnps, pa_index, rho0, c0, p0, alpha, beta, push=False,
        eos=eos, gamma=gamma, kernel=kernel, density_mode='summation'
    )
    result = leapfrog_kick(pa, dt=0.5*dt, dim=nnps.dim, device=nnps.device,
                           push=False)
    if return_dt:
        return result, dt
    return result


def wc_sph_euler_step(nnps, pa_index=0, dt=1.0e-4, rho0=1000.0,
                      c0=20.0, p0=0.0, alpha=0.0, beta=0.0,
                      eos='isothermal', gamma=7.0, kernel='cubic'):
    """Run one minimal WCSPH-style device step.

    The step computes summation density, pressure, optional artificial
    viscosity, and a simple Euler velocity/position update on the device.
    """
    pa = nnps.particles[pa_index]
    compute_summation_density(nnps, pa_index, pa_index, kernel=kernel)
    if eos == 'isothermal':
        compute_isothermal_eos(
            pa, rho0=rho0, c0=c0, p0=p0, device=nnps.device, push=False
        )
        _ensure_sound_speed(pa, c0, nnps.device)
    elif eos == 'tait':
        compute_tait_eos(
            pa, rho0=rho0, c0=c0, gamma=gamma, p0=p0,
            device=nnps.device, push=False
        )
    else:
        raise ValueError("EOS must be 'isothermal' or 'tait'")
    compute_pressure_gradient(
        nnps, pa_index, pa_index, push=False, kernel=kernel
    )
    if alpha != 0.0 or beta != 0.0:
        compute_artificial_viscosity(
            nnps, pa_index, pa_index, alpha=alpha, beta=beta, c0=c0,
            push=False, kernel=kernel
        )
    return euler_step(pa, dt=dt, dim=nnps.dim, device=nnps.device,
                      push=False)
