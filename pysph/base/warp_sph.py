"""Small Warp SPH equation kernels used by the GPU migration prototype."""

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from pysph.base.warp_device_helper import WarpDeviceHelper
from pysph.base.warp_codegen import WarpEquation, build_group_kernel


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
    def _wendland_quintic_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
        # Wendland C2 quintic (PySPH WendlandQuintic), support q < 2.
        # alpha_d = 7/(4 pi) (2D), 21/(16 pi) (3D); dim==1 is unsupported in
        # PySPH and never used here (left as a harmless 2D-base fallback).
        h1 = wp.float64(1.0) / h
        q = rij * h1
        fac = wp.float64(7.0) / (
            wp.float64(4.0) * wp.float64(3.141592653589793)
        )
        if dim == wp.int32(3):
            fac = wp.float64(21.0) / (
                wp.float64(16.0) * wp.float64(3.141592653589793)
            )

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float64(0.0)
        tmp = wp.float64(1.0) - wp.float64(0.5) * q
        if q < wp.float64(2.0):
            val = tmp * tmp * tmp * tmp * (wp.float64(2.0) * q + wp.float64(1.0))
        return val * fac


    @wp.func
    def _wendland_quintic_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
        h1 = wp.float32(1.0) / h
        q = rij * h1
        fac = wp.float32(7.0) / (
            wp.float32(4.0) * wp.float32(3.141592653589793)
        )
        if dim == wp.int32(3):
            fac = wp.float32(21.0) / (
                wp.float32(16.0) * wp.float32(3.141592653589793)
            )

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float32(0.0)
        tmp = wp.float32(1.0) - wp.float32(0.5) * q
        if q < wp.float32(2.0):
            val = tmp * tmp * tmp * tmp * (wp.float32(2.0) * q + wp.float32(1.0))
        return val * fac


    @wp.func
    def _wendland_dwdq_f64(rij: wp.float64, h: wp.float64, dim: wp.int32):
        h1 = wp.float64(1.0) / h
        q = rij * h1
        fac = wp.float64(7.0) / (
            wp.float64(4.0) * wp.float64(3.141592653589793)
        )
        if dim == wp.int32(3):
            fac = wp.float64(21.0) / (
                wp.float64(16.0) * wp.float64(3.141592653589793)
            )

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float64(0.0)
        tmp = wp.float64(1.0) - wp.float64(0.5) * q
        if rij > wp.float64(1.0e-12) and q < wp.float64(2.0):
            val = -wp.float64(5.0) * q * tmp * tmp * tmp
        return val * fac


    @wp.func
    def _wendland_dwdq_f32(rij: wp.float32, h: wp.float32, dim: wp.int32):
        h1 = wp.float32(1.0) / h
        q = rij * h1
        fac = wp.float32(7.0) / (
            wp.float32(4.0) * wp.float32(3.141592653589793)
        )
        if dim == wp.int32(3):
            fac = wp.float32(21.0) / (
                wp.float32(16.0) * wp.float32(3.141592653589793)
            )

        if dim == wp.int32(1):
            fac = fac * h1
        elif dim == wp.int32(2):
            fac = fac * h1 * h1
        else:
            fac = fac * h1 * h1 * h1

        val = wp.float32(0.0)
        tmp = wp.float32(1.0) - wp.float32(0.5) * q
        if rij > wp.float32(1.0e-12) and q < wp.float32(2.0):
            val = -wp.float32(5.0) * q * tmp * tmp * tmp
        return val * fac


    @wp.func
    def _kernel_value_f64(
            rij: wp.float64, h: wp.float64, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_spline_f64(rij, h, dim)
        if kernel_id == wp.int32(2):
            return _wendland_quintic_f64(rij, h, dim)
        return _cubic_spline_f64(rij, h, dim)


    @wp.func
    def _kernel_value_f32(
            rij: wp.float32, h: wp.float32, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_spline_f32(rij, h, dim)
        if kernel_id == wp.int32(2):
            return _wendland_quintic_f32(rij, h, dim)
        return _cubic_spline_f32(rij, h, dim)


    @wp.func
    def _kernel_dwdq_f64(
            rij: wp.float64, h: wp.float64, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_dwdq_f64(rij, h, dim)
        if kernel_id == wp.int32(2):
            return _wendland_dwdq_f64(rij, h, dim)
        return _cubic_dwdq_f64(rij, h, dim)


    @wp.func
    def _kernel_dwdq_f32(
            rij: wp.float32, h: wp.float32, dim: wp.int32,
            kernel_id: wp.int32,
    ):
        if kernel_id == wp.int32(1):
            return _gaussian_dwdq_f32(rij, h, dim)
        if kernel_id == wp.int32(2):
            return _wendland_dwdq_f32(rij, h, dim)
        return _cubic_dwdq_f32(rij, h, dim)


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
    def _tait_eos_hg_correction_f64(
            rho: wp.array(dtype=wp.float64),
            p: wp.array(dtype=wp.float64),
            cs: wp.array(dtype=wp.float64),
            rho0: wp.float64,
            rho01: wp.float64,
            c0: wp.float64,
            gamma: wp.float64,
            gamma1: wp.float64,
            b: wp.float64,
    ):
        # PySPH TaitEOSHGCorrection: clamp wall density to >= rho0 (so wall
        # pressure is always >= 0 and repels), then standard Tait.
        i = wp.tid()
        if rho[i] < rho0:
            rho[i] = rho0
        ratio = rho[i] * rho01
        tmp = wp.pow(ratio, gamma)
        p[i] = b * (tmp - wp.float64(1.0))
        cs[i] = c0 * wp.pow(ratio, gamma1)


    @wp.kernel
    def _tait_eos_hg_correction_f32(
            rho: wp.array(dtype=wp.float32),
            p: wp.array(dtype=wp.float32),
            cs: wp.array(dtype=wp.float32),
            rho0: wp.float32,
            rho01: wp.float32,
            c0: wp.float32,
            gamma: wp.float32,
            gamma1: wp.float32,
            b: wp.float32,
    ):
        i = wp.tid()
        if rho[i] < rho0:
            rho[i] = rho0
        ratio = rho[i] * rho01
        tmp = wp.pow(ratio, gamma)
        p[i] = b * (tmp - wp.float32(1.0))
        cs[i] = c0 * wp.pow(ratio, gamma1)


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
    def _apply_body_force_f64(
            au: wp.array(dtype=wp.float64),
            av: wp.array(dtype=wp.float64),
            aw: wp.array(dtype=wp.float64),
            gx: wp.float64,
            gy: wp.float64,
            gz: wp.float64,
            dim: wp.int32,
    ):
        i = wp.tid()
        au[i] = au[i] + gx
        if dim > wp.int32(1):
            av[i] = av[i] + gy
        if dim > wp.int32(2):
            aw[i] = aw[i] + gz


    @wp.kernel
    def _apply_body_force_f32(
            au: wp.array(dtype=wp.float32),
            av: wp.array(dtype=wp.float32),
            aw: wp.array(dtype=wp.float32),
            gx: wp.float32,
            gy: wp.float32,
            gz: wp.float32,
            dim: wp.int32,
    ):
        i = wp.tid()
        au[i] = au[i] + gx
        if dim > wp.int32(1):
            av[i] = av[i] + gy
        if dim > wp.int32(2):
            aw[i] = aw[i] + gz


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


if wp is not None:
    # Device wp.func objects referenced by generated group kernels. Seeded into
    # the generated kernels' namespace so Warp can resolve them (ADR-0003).
    _WARP_DEVICE_FUNCS = {
        '_kernel_dwdq_f32': _kernel_dwdq_f32,
        '_kernel_dwdq_f64': _kernel_dwdq_f64,
        '_kernel_value_f32': _kernel_value_f32,
        '_kernel_value_f64': _kernel_value_f64,
        '_cubic_spline_f32': _cubic_spline_f32,
        '_cubic_spline_f64': _cubic_spline_f64,
        '_cubic_dwdq_f32': _cubic_dwdq_f32,
        '_cubic_dwdq_f64': _cubic_dwdq_f64,
        '_gaussian_spline_f32': _gaussian_spline_f32,
        '_gaussian_spline_f64': _gaussian_spline_f64,
        '_gaussian_dwdq_f32': _gaussian_dwdq_f32,
        '_gaussian_dwdq_f64': _gaussian_dwdq_f64,
        # Wendland leaves are reachable transitively via the routers'
        # __globals__ (so kernel_id==2 already resolves), but are listed here
        # for parity with cubic/gaussian and robustness against a future
        # generated kernel that calls a leaf directly. Not part of the cache key
        # or any generated source, so this does not perturb the 2D path.
        '_wendland_quintic_f32': _wendland_quintic_f32,
        '_wendland_quintic_f64': _wendland_quintic_f64,
        '_wendland_dwdq_f32': _wendland_dwdq_f32,
        '_wendland_dwdq_f64': _wendland_dwdq_f64,
    }
else:  # pragma: no cover
    _WARP_DEVICE_FUNCS = {}


class ContinuityEquation(WarpEquation):
    """PySPH ``ContinuityEquation`` as a composable Warp block."""
    src_arrays = ('m',)
    out_arrays = ('arho',)
    requires = ('dx', 'dy', 'dz', 'grad', 'vijx', 'vijy', 'vijz')

    def loop(self):
        return (
            "        _acc_arho += s_m[j] * (vijx*(grad*dx) + vijy*(grad*dy)"
            " + vijz*(grad*dz))"
        )


class PressureGradient(WarpEquation):
    """Inviscid WCSPH pressure-gradient acceleration as a Warp block."""
    src_arrays = ('m', 'rho', 'p')
    dst_arrays = ('rho', 'p')
    out_arrays = ('au', 'av', 'aw')
    requires = ('dx', 'dy', 'dz', 'grad')

    def initialize(self):
        return (
            "    rhoi21_ = TYPE(1.0) / (d_rho[i] * d_rho[i])\n"
            "    tmpi_ = d_p[i] * rhoi21_"
        )

    def loop(self):
        return (
            "        rhoj21_ = TYPE(1.0) / (s_rho[j] * s_rho[j])\n"
            "        pg_tmp_ = tmpi_ + s_p[j] * rhoj21_\n"
            "        pg_fac_ = -s_m[j] * pg_tmp_\n"
            "        _acc_au += pg_fac_ * (grad * dx)\n"
            "        _acc_av += pg_fac_ * (grad * dy)\n"
            "        _acc_aw += pg_fac_ * (grad * dz)"
        )


class ArtificialViscosity(WarpEquation):
    """Monaghan artificial viscosity (pair-averaged ``cs``) as a Warp block."""
    src_arrays = ('m', 'rho', 'cs')
    dst_arrays = ('rho', 'cs')
    out_arrays = ('au', 'av', 'aw')
    scalars = ('alpha', 'beta')
    requires = ('dx', 'dy', 'dz', 'rij2', 'hij', 'grad',
                'vijx', 'vijy', 'vijz')

    def loop(self):
        return (
            "        av_vdotx_ = vijx*dx + vijy*dy + vijz*dz\n"
            "        if av_vdotx_ < TYPE(0.0):\n"
            "            av_mu_ = hij * av_vdotx_"
            " / (rij2 + TYPE(0.01)*hij*hij)\n"
            "            av_rhoij1_ = TYPE(2.0) / (d_rho[i] + s_rho[j])\n"
            "            av_cij_ = TYPE(0.5) * (d_cs[i] + s_cs[j])\n"
            "            av_piij_ = (-alpha*av_cij_*av_mu_"
            " + beta*av_mu_*av_mu_) * av_rhoij1_\n"
            "            av_fac_ = -s_m[j] * av_piij_\n"
            "            _acc_au += av_fac_ * grad * dx\n"
            "            _acc_av += av_fac_ * grad * dy\n"
            "            _acc_aw += av_fac_ * grad * dz"
        )


class XSPHCorrection(WarpEquation):
    """PySPH leapfrog XSPH position correction as a Warp block."""
    src_arrays = ('m', 'rho')
    dst_arrays = ('rho',)
    out_arrays = ('ax', 'ay', 'az')
    scalars = ('eps',)
    requires = ('rij', 'hij', 'wij', 'vijx', 'vijy', 'vijz')

    def loop(self):
        return (
            "        xs_rhoij1_ = TYPE(2.0) / (d_rho[i] + s_rho[j])\n"
            "        xs_tmp_ = -eps * s_m[j] * wij * xs_rhoij1_\n"
            "        _acc_ax += xs_tmp_ * vijx\n"
            "        _acc_ay += xs_tmp_ * vijy\n"
            "        _acc_az += xs_tmp_ * vijz"
        )


# The fused continuity-density acceleration group: pressure gradient, then
# Monaghan viscosity (both into au/av/aw), continuity (arho), XSPH (ax/ay/az).
# Block order fixes the per-pair accumulation order for the shared au/av/aw
# accumulators (pressure gradient before viscosity).
_WCSPH_CONTINUITY_BLOCKS = (
    PressureGradient(), ArtificialViscosity(), ContinuityEquation(),
    XSPHCorrection(),
)

# The dam-break fluid acceleration+density group (multi-array): pressure
# gradient, Monaghan viscosity (both -> au/av/aw), continuity (-> arho), FUSED so
# the per-pair geometry (dx/dy/dz, grad, vij, rij2, hij) is computed once per
# neighbour instead of three times. Run once per source array with
# accumulate_outputs=True. Excludes XSPHCorrection -- XSPH sums over the fluid
# only (not the walls), so it stays a separate launch with a different source
# set. Pressure-before-viscosity fixes the au/av/aw accumulation order, matching
# the single-array fused group above.
_WCSPH_DAM_BREAK_FLUID_BLOCKS = (
    PressureGradient(), ArtificialViscosity(), ContinuityEquation(),
)


class SummationDensity(WarpEquation):
    """PySPH ``SummationDensity`` as a composable Warp block."""
    src_arrays = ('m',)
    out_arrays = ('rho',)
    requires = ('rij', 'hij', 'wij')

    def loop(self):
        return "        _acc_rho += s_m[j] * wij"


class WcsphCflFactor(WarpEquation):
    """WCSPH adaptive-timestep per-particle factors as a Warp block.

    ``dt_cfl`` is the neighbor max-reduction of the viscous CFL factor
    ``|hij * (vij . xij) / rij^2| + c0`` (expressed via a free-form ``wp.max``
    accumulation, valid because the factor is non-negative so a zero seed is the
    max identity); ``dt_force`` is the neighbor-independent squared acceleration,
    written in ``post_loop``. Replaces the hand-written ``_wcsph_dt_factors``
    kernels (both flat and grid, via ``neighbor_mode``).
    """
    dst_arrays = ('au', 'av', 'aw')
    out_arrays = ('dt_cfl', 'dt_force')
    scalars = ('c0',)
    requires = ('dx', 'dy', 'dz', 'rij2', 'hij', 'vijx', 'vijy', 'vijz')

    def loop(self):
        return (
            "        if rij2 > TYPE(1.0e-12):\n"
            "            cfl_vdotx_ = vijx*dx + vijy*dy + vijz*dz\n"
            "            cfl_factor_ = wp.abs(hij * cfl_vdotx_ / rij2) + c0\n"
            "            _acc_dt_cfl = wp.max(_acc_dt_cfl, cfl_factor_)"
        )

    def post_loop(self):
        return (
            "    d_dt_force[i] = d_au[i]*d_au[i] + d_av[i]*d_av[i]"
            " + d_aw[i]*d_aw[i]"
        )


def _run_equation_group(nnps, src_index, dst_index, blocks, scalar_values=None,
                        kernel='cubic', cache=None, neighbor_mode='flat',
                        accumulate_outputs=False):
    """Build (or fetch) the generated group kernel for ``blocks`` and launch it.

    Binds device arrays/scalars in the generator's canonical order and writes
    into the destination ``out_arrays``. ``neighbor_mode='grid'`` walks the
    uniform-grid cell list (ignores ``cache``); ``'flat'`` uses the CSR cache
    (built here if ``None``). ``accumulate_outputs=True`` adds to the existing
    destination arrays (read-modify-write). Self- and cross-array (``src !=
    dst``) are both supported because the generator emits ``s_``/``d_`` arrays
    separately. EOS / property / push handling stays with the caller.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for _run_equation_group")
    scalar_values = scalar_values or {}
    src = nnps.particles[src_index].gpu
    dst = nnps.particles[dst_index].gpu
    ndst = dst.get_number_of_particles()
    if ndst <= 0:
        return
    dtype = np.float32 if src.x.dtype == np.float32 else np.float64
    # Periodicity is a property of the NNPS (set_periodic_box): when the grid
    # bounds carry a periodic dimension, build the minimum-image kernel variant
    # and pass the box. Only meaningful in grid mode.
    periodic = bool(
        neighbor_mode == 'grid' and getattr(nnps, '_bounds', None) and (
            nnps._bounds.get('periodic_x') or nnps._bounds.get('periodic_y')
            or nnps._bounds.get('periodic_z')
        )
    )
    group = build_group_kernel(
        blocks, dtype, _WARP_DEVICE_FUNCS, neighbor_mode=neighbor_mode,
        accumulate_outputs=accumulate_outputs, periodic=periodic,
    )
    inputs = [src.get_device_array(n).dev for n in group.src_names]
    inputs += [dst.get_device_array(n).dev for n in group.dst_names]
    if neighbor_mode == 'grid':
        inputs += _grid_launch_args(nnps, src_index, dtype, periodic=periodic)
    else:
        if cache is None:
            cache = nnps.build_neighbor_cache_gpu(src_index, dst_index)
        inputs += [
            cache['starts_dev'], cache['lengths_dev'], cache['neighbors_dev'],
        ]
    inputs += [np.int32(nnps.dim), _kernel_id(kernel)]
    inputs += [dtype(scalar_values[n]) for n in group.scalar_names]
    inputs += [dst.get_device_array(n).dev for n in group.out_names]
    wp.launch(group.kernel, dim=ndst, inputs=inputs, device=nnps.device)
    wp.synchronize_device(nnps.device)


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
        if int(kernel) in (0, 1, 2):
            return np.int32(kernel)
        raise ValueError(
            "kernel id must be 0 (cubic), 1 (gaussian), or 2 (wendland)")
    name = str(kernel).lower().replace('-', '_')
    if name in ('cubic', 'cubic_spline', 'cubicspline'):
        return np.int32(0)
    if name == 'gaussian':
        return np.int32(1)
    if name in ('wendland', 'wendland_quintic', 'wendlandquintic'):
        return np.int32(2)
    raise ValueError("kernel must be 'cubic', 'gaussian', or 'wendland'")


def compute_summation_density(nnps, src_index=0, dst_index=0,
                              out_prop='rho', push=True, kernel='cubic',
                              cache=None, neighbor_mode='flat'):
    """Compute standard SPH summation density with Warp.

    This mirrors ``pysph.sph.basic_equations.SummationDensity`` for one
    source/destination pair using PySPH's standard ``HIJ`` convention:
    ``HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])``.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_summation_density")
    if out_prop != 'rho':
        raise ValueError(
            "compute_summation_density writes the generated SummationDensity "
            "block's canonical 'rho' array; a custom out_prop is not supported."
        )

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_property(dst_pa, out_prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm')
        dst_pa.gpu.push('x', 'y', 'z', 'h', out_prop)
    _run_equation_group(
        nnps, src_index, dst_index, [SummationDensity()],
        kernel=kernel, cache=cache, neighbor_mode=neighbor_mode,
    )
    return dst_pa.gpu.get_device_array(out_prop)


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


def compute_tait_eos_hg_correction(pa, rho0, c0, gamma=7.0, out_prop='p',
                                   cs_prop='cs', device=None, push=True):
    """PySPH ``TaitEOSHGCorrection`` for solid walls (ADR-0005).

    Clamps density to ``>= rho0`` in place (so wall pressure stays ``>= 0`` and
    repels approaching fluid) and then applies Tait EOS for ``p`` and ``cs``.
    Used on boundary/solid arrays in the dam-break step; the fluid uses the
    regular :func:`compute_tait_eos`.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_tait_eos_hg_correction")

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
        kernel = _tait_eos_hg_correction_f32
        rho0 = np.float32(rho0)
        rho01 = np.float32(1.0 / rho0)
        c0 = np.float32(c0)
        gamma = np.float32(gamma)
        gamma1 = np.float32(0.5 * (gamma - np.float32(1.0)))
        b = np.float32(rho0 * c0 * c0 / gamma)
    else:
        kernel = _tait_eos_hg_correction_f64
        rho0 = np.float64(rho0)
        rho01 = np.float64(1.0 / rho0)
        c0 = np.float64(c0)
        gamma = np.float64(gamma)
        gamma1 = np.float64(0.5 * (gamma - np.float64(1.0)))
        b = np.float64(rho0 * c0 * c0 / gamma)
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[rho.dev, out.dev, cs.dev, rho0, rho01, c0, gamma,
                    gamma1, b],
            device=device,
        )
        wp.synchronize_device(device)
    return out, cs


def compute_continuity(nnps, src_index=0, dst_index=0, out_prop='arho',
                       push=True, kernel='cubic', cache=None,
                       neighbor_mode='flat'):
    """Compute PySPH ``ContinuityEquation`` with Warp."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_continuity")
    if out_prop != 'arho':
        raise ValueError(
            "compute_continuity writes the generated ContinuityEquation "
            "block's canonical 'arho' array; a custom out_prop is not supported."
        )

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_property(dst_pa, out_prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'u', 'v', 'w')
        dst_pa.gpu.push('x', 'y', 'z', 'h', 'u', 'v', 'w', out_prop)
    _run_equation_group(
        nnps, src_index, dst_index, [ContinuityEquation()],
        kernel=kernel, cache=cache, neighbor_mode=neighbor_mode,
    )
    return dst_pa.gpu.get_device_array(out_prop)


def compute_pressure_gradient(nnps, src_index=0, dst_index=0,
                              out_props=('au', 'av', 'aw'), push=True,
                              kernel='cubic', cache=None,
                              neighbor_mode='flat'):
    """Compute the inviscid pressure-gradient part of WCSPH momentum."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_pressure_gradient")
    if tuple(out_props) != ('au', 'av', 'aw'):
        raise ValueError(
            "compute_pressure_gradient writes the generated PressureGradient "
            "block's canonical ('au','av','aw'); custom out_props unsupported."
        )

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    for prop in out_props:
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'p')
        dst_pa.gpu.push('x', 'y', 'z', 'h', 'rho', 'p', *out_props)
    _run_equation_group(
        nnps, src_index, dst_index, [PressureGradient()],
        kernel=kernel, cache=cache, neighbor_mode=neighbor_mode,
    )
    dst = dst_pa.gpu
    return (
        dst.get_device_array(out_props[0]),
        dst.get_device_array(out_props[1]),
        dst.get_device_array(out_props[2]),
    )


def compute_artificial_viscosity(nnps, src_index=0, dst_index=0, alpha=0.1,
                                 beta=0.0, c0=20.0,
                                 out_props=('au', 'av', 'aw'), push=True,
                                 kernel='cubic', cache=None,
                                 neighbor_mode='flat'):
    """Add Monaghan artificial viscosity to WCSPH acceleration arrays."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_artificial_viscosity")
    if tuple(out_props) != ('au', 'av', 'aw'):
        raise ValueError(
            "compute_artificial_viscosity writes the generated "
            "ArtificialViscosity block's canonical ('au','av','aw'); custom "
            "out_props unsupported."
        )

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
    # Monaghan viscosity composes onto any prior pressure-gradient
    # acceleration, so the generated group must add to the existing au/av/aw
    # (read-modify-write) rather than overwrite -- matching the hand kernel.
    _run_equation_group(
        nnps, src_index, dst_index, [ArtificialViscosity()],
        scalar_values={'alpha': alpha, 'beta': beta},
        kernel=kernel, cache=cache, neighbor_mode=neighbor_mode,
        accumulate_outputs=True,
    )
    dst = dst_pa.gpu
    return (
        dst.get_device_array(out_props[0]),
        dst.get_device_array(out_props[1]),
        dst.get_device_array(out_props[2]),
    )


def compute_xsph_correction(nnps, src_index=0, dst_index=0, eps=0.5,
                            out_props=('ax', 'ay', 'az'), push=True,
                            kernel='cubic', cache=None,
                            neighbor_mode='flat'):
    """Compute PySPH leapfrog XSPH position correction on the device."""
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for compute_xsph_correction")
    if tuple(out_props) != ('ax', 'ay', 'az'):
        raise ValueError(
            "compute_xsph_correction writes the generated XSPHCorrection "
            "block's canonical ('ax','ay','az'); custom out_props unsupported."
        )

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    for prop in out_props:
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'u', 'v', 'w')
        dst_pa.gpu.push(
            'x', 'y', 'z', 'h', 'rho', 'u', 'v', 'w', *out_props
        )
    _run_equation_group(
        nnps, src_index, dst_index, [XSPHCorrection()],
        scalar_values={'eps': eps}, kernel=kernel, cache=cache,
        neighbor_mode=neighbor_mode,
    )
    dst = dst_pa.gpu
    return (
        dst.get_device_array(out_props[0]),
        dst.get_device_array(out_props[1]),
        dst.get_device_array(out_props[2]),
    )


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


def apply_body_force(pa, gx=0.0, gy=0.0, gz=0.0, dim=3, ramp=1.0,
                     device=None, push=True):
    """Add a (ramped) constant body-force acceleration to ``au``/``av``/``aw``.

    Gravity is a body force, i.e. an acceleration; this adds ``ramp*g`` to the
    acceleration arrays the integrator integrates -- matching PySPH's
    ``MomentumEquation`` ``gz`` term -- so it folds consistently into the PEC
    predictor and corrector. Standalone additive kernel (ADR-0005): it does not
    touch any generated equation kernel or the elliptical-drop step, so with the
    default ``gx=gy=gz=0`` it is a no-op and the 2D path is unchanged. ``ramp``
    in ``[0, 1]`` applies the WCSPH ``n_damp`` gravity startup ramp. Components
    are applied under the same ``dim>1``/``dim>2`` guards as the integrator.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for apply_body_force")

    device = wp.get_device(device)
    _ensure_property(pa, 'au', device)
    _ensure_property(pa, 'av', device)
    _ensure_property(pa, 'aw', device)
    if push:
        pa.gpu.push('au', 'av', 'aw')
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    gx_e, gy_e, gz_e = ramp * gx, ramp * gy, ramp * gz
    if gpu.au.dtype == np.float32:
        kernel = _apply_body_force_f32
        gx_e, gy_e, gz_e = (
            np.float32(gx_e), np.float32(gy_e), np.float32(gz_e)
        )
    else:
        kernel = _apply_body_force_f64
        gx_e, gy_e, gz_e = (
            np.float64(gx_e), np.float64(gy_e), np.float64(gz_e)
        )
    if n > 0:
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                gpu.au.dev, gpu.av.dev, gpu.aw.dev,
                gx_e, gy_e, gz_e, np.int32(dim)
            ],
            device=device,
        )
        wp.synchronize_device(device)
    return gpu.au, gpu.av, gpu.aw


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
                                    dt_min=0.0, dt_max=np.inf, push=True,
                                    cache=None, neighbor_mode='flat'):
    """Compute WCSPH adaptive timestep with device reductions.

    Only the final scalar timestep is copied back to the host. Per-particle
    ``dt_cfl`` and ``dt_force`` remain on the device unless explicitly pulled.

    ``neighbor_mode`` defaults to ``'flat'`` so callers (including the
    summation-density leapfrog path) keep the prebuilt CSR ``cache`` behavior
    unchanged. The continuity-density PEC step passes ``'grid'`` (ADR-0004) to
    walk the uniform-grid cell list directly for the CFL viscous factor and
    avoid building a flat list; in that mode ``cache`` is ignored. Both modes
    visit the same neighbor set, and the CFL factor is an order-independent
    ``max`` reduction, so the resulting timestep matches to fp32 scale.
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
    if cache is None and neighbor_mode == 'flat':
        cache = nnps.build_neighbor_cache_gpu(pa_index, pa_index)
    gpu = pa.gpu
    n = gpu.get_number_of_particles()
    dt_cfl = gpu.get_device_array('dt_cfl')
    dt_force = gpu.get_device_array('dt_force')
    if gpu.x.dtype == np.float32:
        dtype = wp.float32
        init_kernel = _wcsph_dt_init_f32
        reduce_kernel = _wcsph_dt_reduce_f32
        finalize_kernel = _wcsph_dt_finalize_f32
        cfl = np.float32(cfl)
        dt_min = np.float32(dt_min)
        dt_max = np.float32(dt_max)
    else:
        dtype = wp.float64
        init_kernel = _wcsph_dt_init_f64
        reduce_kernel = _wcsph_dt_reduce_f64
        finalize_kernel = _wcsph_dt_finalize_f64
        cfl = np.float64(cfl)
        dt_min = np.float64(dt_min)
        dt_max = np.float64(dt_max)

    max_cfl = wp.zeros(1, dtype=dtype, device=nnps.device)
    max_force = wp.zeros(1, dtype=dtype, device=nnps.device)
    min_h = wp.zeros(1, dtype=dtype, device=nnps.device)
    out_dt = wp.zeros(1, dtype=dtype, device=nnps.device)
    if n > 0:
        # Per-particle CFL/force factors via the generated WcsphCflFactor group
        # (flat or grid). dt_cfl is a neighbor max-reduction, dt_force a
        # per-particle term; both written into pa's dt_cfl/dt_force arrays.
        _run_equation_group(
            nnps, pa_index, pa_index, [WcsphCflFactor()],
            scalar_values={'c0': c0}, cache=cache, neighbor_mode=neighbor_mode,
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


def _apply_wcsph_eos(nnps, pa, rho0, c0, p0, eos, gamma):
    """Apply the equation of state (per-particle, no neighbor loop).

    Shared by the summation-density path and the fused continuity path.
    """
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


def _grid_launch_args(nnps, src_index, dtype, periodic=False):
    """Ordered grid-query launch inputs for a grid-direct kernel (ADR-0004).

    Mirrors the signature emitted by ``warp_codegen`` in ``grid`` mode: the
    device cell list from ``_build_grid`` (reused per ``update()``) followed by
    the grid bounds and ``radius_scale``. When ``periodic`` is set, the periodic
    box lengths and per-dimension periodic flags are appended, matching the
    minimum-image kernel variant.
    """
    grid = nnps._build_grid(src_index)
    b = nnps._bounds
    args = [
        grid['starts'], grid['counts'], grid['cell_particles'],
        dtype(b['xmin']), dtype(b['ymin']), dtype(b['zmin']),
        dtype(nnps.cell_size),
        np.int32(b['nx']), np.int32(b['ny']), np.int32(b['nz']),
        np.int32(b['ncells']),
        dtype(nnps.radius_scale),
    ]
    if periodic:
        args += [
            dtype(b['box_lx']), dtype(b['box_ly']), dtype(b['box_lz']),
            np.int32(1 if b['periodic_x'] else 0),
            np.int32(1 if b['periodic_y'] else 0),
            np.int32(1 if b['periodic_z'] else 0),
        ]
    return args


def compute_wcsph_accel_continuity(nnps, src_index=0, dst_index=0, alpha=0.1,
                                   beta=0.0, eps=0.5, c0=20.0, kernel='cubic',
                                   cache=None, push=True, neighbor_mode='grid'):
    """Fused continuity-density acceleration via a generated group kernel.

    One neighbor traversal computes ``ContinuityEquation`` (``arho``), the
    inviscid pressure gradient plus Monaghan artificial viscosity
    (``au, av, aw``), and the XSPH correction (``ax, ay, az``), replacing four
    separate launches over the same neighbor cache (ADR-0003). EOS must have
    been applied beforehand because the kernel reads ``p`` and ``cs``.

    ``neighbor_mode='grid'`` (default, ADR-0004) walks the uniform-grid cell
    list directly and ignores ``cache``, so no flat neighbor list is built;
    ``'flat'`` reads the prebuilt CSR ``cache`` (built here if ``None``) and is
    retained for the oracle/host-query path and parity tests.
    """
    if wp is None:  # pragma: no cover
        raise ImportError(
            "warp is required for compute_wcsph_accel_continuity"
        )

    src_pa = nnps.particles[src_index]
    dst_pa = nnps.particles[dst_index]
    _ensure_sound_speed(src_pa, c0, nnps.device)
    if dst_pa is not src_pa:
        _ensure_sound_speed(dst_pa, c0, nnps.device)
    for prop in ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'):
        _ensure_property(dst_pa, prop, nnps.device)

    if push:
        src_pa.gpu.push(
            'x', 'y', 'z', 'h', 'm', 'rho', 'p', 'cs', 'u', 'v', 'w'
        )
        dst_pa.gpu.push(
            'x', 'y', 'z', 'h', 'rho', 'p', 'cs', 'u', 'v', 'w',
            'au', 'av', 'aw', 'arho', 'ax', 'ay', 'az'
        )
    dst = dst_pa.gpu
    if dst.get_number_of_particles() <= 0:
        return None
    _run_equation_group(
        nnps, src_index, dst_index, _WCSPH_CONTINUITY_BLOCKS,
        scalar_values={'alpha': alpha, 'beta': beta, 'eps': eps},
        kernel=kernel, cache=cache, neighbor_mode=neighbor_mode,
    )
    return (
        dst.get_device_array('au'), dst.get_device_array('av'),
        dst.get_device_array('aw'),
    )


def _compute_wcsph_acceleration(nnps, pa_index, rho0, c0, p0, alpha, beta,
                                push, eos, gamma, kernel,
                                density_mode='summation', cache=None):
    # Grid-direct (ADR-0004 extended): all neighbor consumers walk the cell list
    # directly, so the summation step builds no flat CSR neighbor cache. The
    # pressure-gradient(overwrite) -> viscosity(add) composition is unchanged;
    # only the neighbor source (and thus fp32 visitation order) differs.
    pa = nnps.particles[pa_index]
    if density_mode == 'summation':
        compute_summation_density(
            nnps, pa_index, pa_index, push=push, kernel=kernel,
            neighbor_mode='grid'
        )
    elif density_mode == 'continuity':
        _ensure_property(pa, 'arho', nnps.device)
        if push:
            pa.gpu.push('rho', 'arho')
    else:
        raise ValueError("density_mode must be 'summation' or 'continuity'")
    _apply_wcsph_eos(nnps, pa, rho0, c0, p0, eos, gamma)
    result = compute_pressure_gradient(
        nnps, pa_index, pa_index, push=False, kernel=kernel,
        neighbor_mode='grid'
    )
    if alpha != 0.0 or beta != 0.0:
        result = compute_artificial_viscosity(
            nnps, pa_index, pa_index, alpha=alpha, beta=beta, c0=c0,
            push=False, kernel=kernel, neighbor_mode='grid'
        )
    if density_mode == 'continuity':
        compute_continuity(
            nnps, pa_index, pa_index, push=False, kernel=kernel,
            neighbor_mode='grid'
        )
    return result


def _wc_sph_pec_continuity_step(nnps, pa_index, dt, rho0, c0, p0,
                                periodic_bounds, push, alpha, beta, eos,
                                gamma, kernel, xsph_eps, adaptive_dt, cfl,
                                dt_min, dt_max, adaptive_dt_scale,
                                step_dt_max):
    pa = nnps.particles[pa_index]
    use_xsph = xsph_eps is not None and xsph_eps != 0.0
    eps = 0.0 if xsph_eps is None else xsph_eps
    if push:
        nnps.update(push=True)
    save_wcsph_state(pa, dim=nnps.dim, device=nnps.device, push=push)

    # ADR-0004: both neighbor consumers walk the cell list directly, so the
    # grid is the only spatial index built per half-stage (cached per update);
    # no flat CSR neighbor list is materialized on the continuity hot path.
    _ensure_property(pa, 'arho', nnps.device)
    _apply_wcsph_eos(nnps, pa, rho0, c0, p0, eos, gamma)
    compute_wcsph_accel_continuity(
        nnps, pa_index, pa_index, alpha=alpha, beta=beta, eps=eps, c0=c0,
        kernel=kernel, push=False
    )
    if adaptive_dt:
        dt = compute_wcsph_adaptive_timestep(
            nnps, pa_index=pa_index, c0=c0, cfl=cfl, dt_min=dt_min,
            dt_max=dt_max, push=False, neighbor_mode='grid'
        )
        dt = min(float(dt) * float(adaptive_dt_scale), float(step_dt_max))
    wcsph_pec_stage(
        pa, dt=dt, stage=0.5, dim=nnps.dim, xsph=use_xsph,
        device=nnps.device, push=False
    )
    wrap_periodic(pa, periodic_bounds, dim=nnps.dim, device=nnps.device)
    nnps.update(push=False)

    _apply_wcsph_eos(nnps, pa, rho0, c0, p0, eos, gamma)
    compute_wcsph_accel_continuity(
        nnps, pa_index, pa_index, alpha=alpha, beta=beta, eps=eps, c0=c0,
        kernel=kernel, push=False
    )
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
                         density_mode='summation', adaptive_dt_scale=1.0,
                         step_dt_max=np.inf):
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
            dt_min, dt_max, adaptive_dt_scale, step_dt_max
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
            dt_max=dt_max, push=False, neighbor_mode='grid'
        )
        dt = min(float(dt) * float(adaptive_dt_scale), float(step_dt_max))
    leapfrog_kick(pa, dt=0.5*dt, dim=nnps.dim, device=nnps.device,
                  push=False)
    if xsph_eps is None or xsph_eps == 0.0:
        leapfrog_drift(
            pa, dt=dt, dim=nnps.dim, device=nnps.device, push=False
        )
    else:
        compute_xsph_correction(
            nnps, pa_index, pa_index, eps=xsph_eps, push=False,
            kernel=kernel, neighbor_mode='grid'
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


def _zero_device_props(pa, props):
    """Zero the named device arrays in place (no host transfer)."""
    for prop in props:
        pa.gpu.get_device_array(prop).dev.zero_()


def wc_sph_dam_break_step(nnps, fluid_index=0, solid_indices=(1,), dt=1.0e-4,
                          rho0=1000.0, c0=10.0, p0=0.0, alpha=0.1, beta=0.0,
                          gamma=7.0, kernel='wendland', xsph_eps=0.5,
                          gx=0.0, gy=0.0, gz=-9.81, gravity_ramp=1.0,
                          adaptive_dt=False, cfl=0.25, dt_min=0.0,
                          dt_max=np.inf, adaptive_dt_scale=1.0,
                          step_dt_max=np.inf, push=False, return_dt=False):
    """One 3D dam-break WCSPH continuity-density PEC step (ADR-0005).

    Multi-array: the fluid's acceleration and density rate sum over the fluid
    plus every solid wall array; each wall integrates density from the fluid
    only and is otherwise fixed -- walls start at rest with zero acceleration,
    so the shared PEC stage leaves their position/velocity unchanged while their
    density (hence pressure) responds to approaching fluid. Walls use
    ``TaitEOSHGCorrection`` (clamped ``p >= 0``); the fluid uses Tait EOS.
    Gravity is added to the fluid acceleration with an optional ``n_damp`` ramp
    (``gravity_ramp`` in ``[0, 1]``). All neighbour traversal is grid-direct.

    This is additive to the backend: it composes the existing generated equation
    blocks (run with ``accumulate_outputs=True`` over each source) and the
    existing PEC stage; it does not modify any single-array path, so the 2D
    elliptical-drop step is unchanged.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for wc_sph_dam_break_step")

    fluid_index = int(fluid_index)
    solid_indices = [int(i) for i in solid_indices]
    fluid = nnps.particles[fluid_index]
    solids = [nnps.particles[i] for i in solid_indices]
    arrays = [fluid] + solids
    sources_for_fluid = [fluid_index] + solid_indices
    use_xsph = xsph_eps is not None and xsph_eps != 0.0
    eps = 0.0 if xsph_eps is None else xsph_eps
    dim = nnps.dim
    device = nnps.device

    out_props = ('au', 'av', 'aw', 'arho', 'ax', 'ay', 'az')
    for pa in arrays:
        for prop in ('rho', 'p', 'cs') + out_props:
            _ensure_property(pa, prop, device)
    if push:
        for pa in arrays:
            pa.gpu.push('x', 'y', 'z', 'h', 'm', 'rho', 'p', 'cs',
                        'u', 'v', 'w', *out_props)
    nnps.update(push=push)

    for pa in arrays:
        save_wcsph_state(pa, dim=dim, device=device, push=False)

    def accel():
        # EOS: fluid Tait; walls Tait-HG (clamp rho>=rho0 so wall p>=0).
        compute_tait_eos(fluid, rho0=rho0, c0=c0, gamma=gamma, p0=p0,
                         device=device, push=False)
        for s in solids:
            compute_tait_eos_hg_correction(s, rho0=rho0, c0=c0, gamma=gamma,
                                            device=device, push=False)
        # Zero accumulators (walls keep zero accel -> they stay fixed).
        for pa in arrays:
            _zero_device_props(pa, out_props)
        # Fluid: pressure + Monaghan AV + continuity summed over fluid + walls,
        # fused into ONE kernel per source (single neighbour walk / single
        # per-pair geometry for all three blocks instead of three).
        for s_index in sources_for_fluid:
            _run_equation_group(nnps, s_index, fluid_index,
                                list(_WCSPH_DAM_BREAK_FLUID_BLOCKS),
                                scalar_values={'alpha': alpha, 'beta': beta},
                                kernel=kernel, neighbor_mode='grid',
                                accumulate_outputs=True)
        # XSPH position correction from fluid neighbours only.
        if use_xsph:
            _run_equation_group(nnps, fluid_index, fluid_index,
                                [XSPHCorrection()],
                                scalar_values={'eps': eps}, kernel=kernel,
                                neighbor_mode='grid', accumulate_outputs=True)
        # Walls: density rate from the fluid only.
        for w_index in solid_indices:
            _run_equation_group(nnps, fluid_index, w_index,
                                [ContinuityEquation()], kernel=kernel,
                                neighbor_mode='grid', accumulate_outputs=True)
        # Gravity (ramped) into the fluid acceleration.
        apply_body_force(fluid, gx=gx, gy=gy, gz=gz, dim=dim,
                         ramp=gravity_ramp, device=device, push=False)

    # Predictor half-stage (dt fixed for both stages, set adaptively here).
    accel()
    if adaptive_dt:
        dt = compute_wcsph_adaptive_timestep(
            nnps, pa_index=fluid_index, c0=c0, cfl=cfl, dt_min=dt_min,
            dt_max=dt_max, push=False, neighbor_mode='grid'
        )
        dt = min(float(dt) * float(adaptive_dt_scale), float(step_dt_max))
    for pa in arrays:
        wcsph_pec_stage(pa, dt=dt, stage=0.5, dim=dim,
                        xsph=(use_xsph and pa is fluid), device=device,
                        push=False)
    nnps.update(push=False)

    # Corrector half-stage.
    accel()
    for pa in arrays:
        wcsph_pec_stage(pa, dt=dt, stage=1.0, dim=dim,
                        xsph=(use_xsph and pa is fluid), device=device,
                        push=False)
    nnps.update(push=False)

    if return_dt:
        return dt
    return dt


def wc_sph_euler_step(nnps, pa_index=0, dt=1.0e-4, rho0=1000.0,
                      c0=20.0, p0=0.0, alpha=0.0, beta=0.0,
                      eos='isothermal', gamma=7.0, kernel='cubic'):
    """Run one minimal WCSPH-style device step.

    The step computes summation density, pressure, optional artificial
    viscosity, and a simple Euler velocity/position update on the device.
    """
    pa = nnps.particles[pa_index]
    compute_summation_density(
        nnps, pa_index, pa_index, kernel=kernel, neighbor_mode='grid'
    )
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
        nnps, pa_index, pa_index, push=False, kernel=kernel,
        neighbor_mode='grid'
    )
    if alpha != 0.0 or beta != 0.0:
        compute_artificial_viscosity(
            nnps, pa_index, pa_index, alpha=alpha, beta=beta, c0=c0,
            push=False, kernel=kernel, neighbor_mode='grid'
        )
    return euler_step(pa, dt=dt, dim=nnps.dim, device=nnps.device,
                      push=False)
