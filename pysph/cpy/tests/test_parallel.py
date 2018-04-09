from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config
from ..array import wrap
from ..types import annotate
from ..parallel import Elementwise, Reduction, Kernel, LocalMem


class TestParallelUtils(unittest.TestCase):

    def setUp(self):
        cfg = get_config()
        self._use_double = cfg.use_double
        cfg.use_double = True

    def tearDown(self):
        get_config().use_double = self._use_double

    def _check_simple_elementwise(self, backend):
        # Given
        def axpb(i=0, x=[0.0], y=[0.0], a=1.0, b=1.0):
            y[i] = a*sin(x[i]) + b

        x = np.linspace(0, 1, 10000)
        y = np.zeros_like(x)
        a = 2.0
        b = 3.0
        x, y = wrap(x, y, backend=backend)

        # When
        e = Elementwise(axpb, backend=backend)
        e(x, y, a, b)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, a*np.sin(x.data) + b))

    def test_elementwise_works_with_cython(self):
        self._check_simple_elementwise(backend='cython')

    def test_elementwise_works_with_opencl(self):
        importorskip('pyopencl')

        self._check_simple_elementwise(backend='opencl')

    def _check_simple_reduction(self, backend):
        x = np.linspace(0, 1, 1000)/1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('a+b', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def _check_reduction_min(self, backend):
        x = np.linspace(0, 1, 1000)/1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('min(a, b)', neutral='INFINITY', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.0, 6)

    def _check_reduction_with_map(self, backend):
        # Given
        from math import cos, sin
        x = np.linspace(0, 1, 1000)/1000
        y = x.copy()
        x, y = wrap(x, y, backend=backend)

        def map(i=0, x=[0.0], y=[0.0]):
            return cos(x[i])*sin(y[i])

        # When
        r = Reduction('a+b', map_func=map, backend=backend)
        result = r(x, y)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def test_reduction_works_without_map_cython(self):
        self._check_simple_reduction(backend='cython')

    def test_reduction_works_with_map_cython(self):
        self._check_reduction_with_map(backend='cython')

    def test_reduction_works_neutral_cython(self):
        self._check_reduction_min(backend='cython')

    def test_reduction_works_without_map_opencl(self):
        importorskip('pyopencl')
        self._check_simple_reduction(backend='opencl')

    def test_reduction_works_with_map_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_with_map(backend='opencl')

    def test_reduction_works_neutral_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_min(backend='opencl')


class TestKernel(unittest.TestCase):

    def setUp(self):
        importorskip('pyopencl')

    def test_simple_kernel(self):
        # Given
        @annotate(gdoublep='x, y', a='float')
        def knl(x, y, a):
            i = declare('int')
            i = GID_0*LDIM_0 + LID_0
            y[i] = x[i]*a

        x = np.linspace(0, 1, 1000)
        y = np.zeros_like(x)
        x, y = wrap(x, y, backend='opencl')

        # When
        k = Kernel(knl, backend='opencl')
        a = 21.0
        k(x, y, a)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data*a))

    def test_kernel_with_local_memory(self):
        # Given
        @annotate(gdoublep='x, y', xc='ldoublep', a='float')
        def knl(x, y, xc, a):
            i, lid = declare('int', 2)
            lid = LID_0
            i = GID_0*LDIM_0 + lid

            xc[lid] = x[i]

            local_barrier()

            y[i] = xc[lid]*a

        x = np.linspace(0, 1, 1024)
        y = np.zeros_like(x)
        xc = LocalMem(1, backend='opencl')

        x, y = wrap(x, y, backend='opencl')

        # When
        k = Kernel(knl, backend='opencl')
        a = 21.0
        k(x, y, xc, a)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data*a))
