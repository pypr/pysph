import numpy as np
try:
    from scipy.integrate import quad
except ImportError:
    quad = None
from unittest import TestCase, main

from pysph.base.kernels import CubicSpline, get_compiled_kernel


###############################################################################
# `TestKernelBase` class.
###############################################################################
class TestKernelBase(TestCase):
    """Base class for all kernel tests.
    """
    kernel_factory = None
    @classmethod
    def setUpClass(cls):
        cls.wrapper = get_compiled_kernel(cls.kernel_factory())
        cls.kernel = cls.wrapper.kernel
        cls.gradient = cls.wrapper.gradient

    def setUp(self):
        self.kernel = self.__class__.kernel
        self.gradient = self.__class__.gradient

    def check_kernel_moment_1d(self, a, b, h, m, xj=0.0):
        func = self.kernel
        if m == 0:
            f = lambda x: func(x, 0, 0, xj, 0, 0, h)
        else:
            f = lambda x: pow(x, m)*func(x, 0, 0, xj, 0, 0, h)
        if quad is None:
            kern_f = np.vectorize(f)
            nx = 201
            x = np.linspace(a, b, nx)
            result = np.sum(kern_f(x))*(b-a)/(nx-1)
        else:
            result = quad(f, a, b)[0]

        return result

    def check_grad_moment_1d(self, a, b, h, m, xj=0.0):
        func = self.gradient
        if m == 0:
            f = lambda x: func(x, 0, 0, xj, 0, 0, h)[0]
        else:
            f = lambda x: pow(x-xj, m)*func(x, 0, 0, xj, 0, 0, h)[0]
        if quad is None:
            kern_f = np.vectorize(f)
            nx = 201
            x = np.linspace(a, b, nx)
            return np.sum(kern_f(x))*(b-a)/(nx-1)
        else:
            return quad(f, a, b)[0]

    def check_kernel_moment_2d(self, m, n):
        x0, y0, z0 = 0.5, 0.5, 0.0
        def func(x, y):
            fac = pow(x-x0, m)*pow(y-y0, n)
            return fac*self.kernel(x, y, 0.0, x0, y0, 0.0, 0.15)
        vfunc = np.vectorize(func)
        nx, ny = 101, 101
        vol = 1.0/(nx-1)*1.0/(ny-1)
        x, y = np.mgrid[0:1:nx*1j, 0:1:nx*1j]
        result = np.sum(vfunc(x, y))*vol
        return result

    def check_gradient_moment_2d(self, m, n):
        x0, y0, z0 = 0.5, 0.5, 0.0
        def func(x, y):
            fac = pow(x-x0, m)*pow(y-y0, n)
            return fac*np.asarray(self.gradient(x, y, 0.0, x0, y0, 0.0, 0.15))
        vfunc = np.vectorize(func, otypes=[np.ndarray])
        nx, ny = 101, 101
        vol = 1.0/(nx-1)*1.0/(ny-1)
        x, y = np.mgrid[0:1:nx*1j, 0:1:nx*1j]
        result = np.sum(vfunc(x, y))*vol
        return result

    def check_kernel_moment_3d(self, l, m, n):
        x0, y0, z0 = 0.5, 0.5, 0.5
        def func(x, y, z):
            fac = pow(x-x0, l)*pow(y-y0, m)*pow(z-z0, n)
            return fac*self.kernel(x, y, z, x0, y0, z0, 0.15)
        vfunc = np.vectorize(func)
        nx, ny, nz = 51, 51, 51
        vol = 1.0/(nx-1)*1.0/(ny-1)*1.0/(nz-1)
        x, y, z = np.mgrid[0:1:nx*1j, 0:1:ny*1j, 0:1:nz*1j]
        result = np.sum(vfunc(x, y, z))*vol
        return result

    def check_gradient_moment_3d(self, l, m, n):
        x0, y0, z0 = 0.5, 0.5, 0.5
        def func(x, y, z):
            fac = pow(x-x0, l)*pow(y-y0, m)*pow(z-z0, n)
            return fac*np.asarray(self.gradient(x, y, z, x0, y0, z0, 0.15))
        vfunc = np.vectorize(func, otypes=[np.ndarray])
        nx, ny, nz = 51, 51, 51
        vol = 1.0/(nx-1)*1.0/(ny-1)*1.0/(nz-1)
        x, y, z = np.mgrid[0:1:nx*1j, 0:1:ny*1j, 0:1:nz*1j]
        result = np.sum(vfunc(x, y, z))*vol
        return result

###############################################################################
# `TestCubicSpline1D` class.
###############################################################################
class TestCubicSpline1D(TestKernelBase):
    kernel_factory = staticmethod(lambda: CubicSpline(dim=1))

    def test_simple(self):
        k = self.kernel(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 2./3
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))
        k = self.kernel(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))

        g = self.gradient(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))
        g = self.gradient(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))

    def test_zeroth_kernel_moments(self):
        # zero'th moment
        r = self.check_kernel_moment_1d(-2.0, 2.0, 1.0, 0, xj=0)
        self.assertAlmostEqual(r, 1.0, 8)
        # Use a non-unit h.
        r = self.check_kernel_moment_1d(-2.0, 2.0, 0.5, 0, xj=0)
        self.assertAlmostEqual(r, 1.0, 8)
        r = self.check_kernel_moment_1d(0.0, 4.0, 1.0, 0, xj=2.0)
        self.assertAlmostEqual(r, 1.0, 8)

    def test_first_kernel_moment(self):
        r = self.check_kernel_moment_1d(-2.0, 2.0, 1.0, 1, xj=0.0)
        self.assertAlmostEqual(r, 0.0, 8)

    def test_zeroth_grad_moments(self):
        # zero'th moment
        r = self.check_grad_moment_1d(-2.0, 2.0, 1.0, 0, xj=0)
        self.assertAlmostEqual(r, 0.0, 8)
        # Use a non-unit h.
        r = self.check_grad_moment_1d(-2.0, 2.0, 0.5, 0, xj=0)
        self.assertAlmostEqual(r, 0.0, 8)
        r = self.check_grad_moment_1d(0.0, 4.0, 1.0, 0, xj=2.0)
        self.assertAlmostEqual(r, 0.0, 8)

    def test_first_grad_moment(self):
        r = self.check_grad_moment_1d(0.0, 4.0, 1.0, 1, xj=2.0)
        self.assertAlmostEqual(r, -1.0, 8)

###############################################################################
# `TestCubicSpline2D` class.
###############################################################################
class TestCubicSpline2D(TestKernelBase):
    kernel_factory = staticmethod(lambda: CubicSpline(dim=2))

    def test_simple(self):
        k = self.kernel(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 10./(7*np.pi)
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))
        k = self.kernel(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))

        g = self.gradient(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))
        g = self.gradient(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))

    def test_zeroth_kernel_moments(self):
        r = self.check_kernel_moment_2d(0, 0)
        self.assertAlmostEqual(r, 1.0, 7)

    def test_first_kernel_moment(self):
        r = self.check_kernel_moment_2d(0, 1)
        self.assertAlmostEqual(r, 0.0, 7)
        r = self.check_kernel_moment_2d(1, 0)
        self.assertAlmostEqual(r, 0.0, 7)
        r = self.check_kernel_moment_2d(1, 1)
        self.assertAlmostEqual(r, 0.0, 7)

    def test_zeroth_grad_moments(self):
        r = self.check_gradient_moment_2d(0, 0)
        self.assertAlmostEqual(r[0], 0.0, 7)
        self.assertAlmostEqual(r[1], 0.0, 7)

    def test_first_grad_moment(self):
        r = self.check_gradient_moment_2d(1, 0)
        self.assertAlmostEqual(r[0], -1.0, 6)
        self.assertAlmostEqual(r[1], 0.0, 8)
        r = self.check_gradient_moment_2d(0, 1)
        self.assertAlmostEqual(r[0], 0.0, 8)
        self.assertAlmostEqual(r[1], -1.0, 6)
        r = self.check_gradient_moment_2d(1, 1)
        self.assertAlmostEqual(r[0], 0.0, 8)
        self.assertAlmostEqual(r[1], 0.0, 8)


###############################################################################
# `TestCubicSpline3D` class.
###############################################################################
class TestCubicSpline3D(TestKernelBase):
    kernel_factory = staticmethod(lambda: CubicSpline(dim=3))

    def test_simple(self):
        k = self.kernel(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 1./np.pi
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))
        k = self.kernel(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(k, expect,
                               msg='Kernel value %s != %s (expected)'%(k, expect))

        g = self.gradient(xi=0.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))
        g = self.gradient(xi=3.0, yi=0.0, zi=0.0, xj=0.0, yj=0.0, zj=0.0, h=1.0)
        expect = 0.0
        self.assertAlmostEqual(g[0], expect,
                               msg='Kernel value %s != %s (expected)'%(g[0], expect))

    def test_zeroth_kernel_moments(self):
        r = self.check_kernel_moment_3d(0, 0, 0)
        self.assertAlmostEqual(r, 1.0, 6)

    def test_first_kernel_moment(self):
        r = self.check_kernel_moment_3d(0, 0, 1)
        self.assertAlmostEqual(r, 0.0, 7)
        r = self.check_kernel_moment_3d(0, 1, 0)
        self.assertAlmostEqual(r, 0.0, 7)
        r = self.check_kernel_moment_3d(1, 0, 1)
        self.assertAlmostEqual(r, 0.0, 7)

    def test_zeroth_grad_moments(self):
        r = self.check_gradient_moment_3d(0, 0, 0)
        self.assertAlmostEqual(r[0], 0.0, 7)
        self.assertAlmostEqual(r[1], 0.0, 7)
        self.assertAlmostEqual(r[2], 0.0, 7)

    def test_first_grad_moment(self):
        r = self.check_gradient_moment_3d(1, 0, 0)
        self.assertAlmostEqual(r[0], -1.0, 4)
        self.assertAlmostEqual(r[1], 0.0, 8)
        self.assertAlmostEqual(r[2], 0.0, 8)
        r = self.check_gradient_moment_3d(0, 1, 0)
        self.assertAlmostEqual(r[0], 0.0, 8)
        self.assertAlmostEqual(r[1], -1.0, 4)
        self.assertAlmostEqual(r[2], 0.0, 6)
        r = self.check_gradient_moment_3d(0, 0, 1)
        self.assertAlmostEqual(r[0], 0.0, 8)
        self.assertAlmostEqual(r[1], 0.0, 8)
        self.assertAlmostEqual(r[2], -1.0, 4)

if __name__ == '__main__':
    main()
