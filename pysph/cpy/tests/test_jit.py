from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap
from ..types import annotate
from ..jit import ElementwiseJIT, ReductionJIT


class TestJIT(unittest.TestCase):
    def setUp(self):
        cfg = get_config()
        self._use_double = cfg.use_double
        cfg.use_double = True

    def tearDown(self):
        get_config().use_double = self._use_double

    def _check_simple_elementwise_jit(self, backend):
        # Given
        def axpb(i, x, y, a, b):
            y[i] = a * sin(x[i]) + b

        x = np.linspace(0, 1, 10000)
        y = np.zeros_like(x)
        a = 2.0
        b = 3.0
        x, y = wrap(x, y, backend=backend)

        # When
        e = ElementwiseJIT(axpb, backend=backend)
        e(x, y, a, b)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, a * np.sin(x.data) + b))

    def test_elementwise_jit_works_with_cython(self):
        self._check_simple_elementwise_jit(backend='cython')

    def test_elementwise_jit_works_with_opencl(self):
        importorskip('pyopencl')

        self._check_simple_elementwise_jit(backend='opencl')

    def test_elementwise_jit_works_with_cuda(self):
        importorskip('pycuda')

        self._check_simple_elementwise_jit(backend='cuda')

    def _check_simple_reduction_jit(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = ReductionJIT('a+b', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def _check_reduction_min_jit(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = ReductionJIT('min(a, b)', neutral='INFINITY', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.0, 6)

    def _check_reduction_with_map_jit(self, backend):
        # Given
        from math import cos, sin
        x = np.linspace(0, 1, 1000) / 1000
        y = x.copy()
        x, y = wrap(x, y, backend=backend)

        def map(i=0, x=[0.0], y=[0.0]):
            return cos(x[i]) * sin(y[i])

        # When
        r = ReductionJIT('a+b', map_func=map, backend=backend)
        result = r(x, y)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def test_reduction_jit_works_without_map_cython(self):
        self._check_simple_reduction_jit(backend='cython')

    def test_reduction_jit_works_with_map_cython(self):
        self._check_reduction_with_map_jit(backend='cython')

    def test_reduction_jit_works_neutral_cython(self):
        self._check_reduction_min_jit(backend='cython')

    def test_reduction_jit_works_without_map_opencl(self):
        importorskip('pyopencl')
        self._check_simple_reduction_jit(backend='opencl')

    def test_reduction_jit_works_with_map_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_with_map_jit(backend='opencl')

    def test_reduction_jit_works_neutral_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_min_jit(backend='opencl')

    def test_reduction_jit_works_without_map_cuda(self):
        importorskip('pycuda')
        self._check_simple_reduction_jit(backend='cuda')

    def test_reduction_jit_works_with_map_cuda(self):
        importorskip('pycuda')
        self._check_reduction_with_map_jit(backend='cuda')

    def test_reduction_jit_works_neutral_cuda(self):
        importorskip('pycuda')
        self._check_reduction_min_jit(backend='cuda')


