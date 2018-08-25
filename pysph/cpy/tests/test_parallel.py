from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config
from ..array import wrap
from ..types import annotate
from ..parallel import Elementwise, Reduction, Scan


class TestParallelUtils(unittest.TestCase):

    def setUp(self):
        cfg = get_config()
        self._use_double = cfg.use_double
        cfg.use_double = True

    def tearDown(self):
        get_config().use_double = self._use_double

    def _check_simple_elementwise(self, backend):
        # Given
        @annotate(i='int', x='doublep', y='doublep', double='a,b')
        def axpb(i, x, y, a, b):
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

    def test_elementwise_works_with_cuda(self):
        importorskip('pycuda')

        self._check_simple_elementwise(backend='cuda')

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

        @annotate(i='int', doublep='x, y')
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

    def test_reduction_works_without_map_cuda(self):
        importorskip('pycuda')
        self._check_simple_reduction(backend='cuda')

    def test_reduction_works_with_map_cuda(self):
        importorskip('pycuda')
        self._check_reduction_with_map(backend='cuda')

    def test_reduction_works_neutral_cuda(self):
        importorskip('pycuda')
        self._check_reduction_min(backend='cuda')

    def test_scan_works_opencl(self):
        importorskip('pyopencl')
        # Given
        a = np.arange(10000, dtype=np.int32)
        data = a.copy()
        a = wrap(a, backend='opencl')

        @annotate(i='int', ary='intp', return_='int')
        def input_f(i, ary):
            return ary[i]

        @annotate(int='i, item', ary='intp')
        def output_f(i, ary, item):
            ary[i+1] = item

        # When
        s = Scan(input_f, output_f, 'a+b', dtype=np.int32, backend='opencl')
        s(a)
        a.pull()
        result = a.data

        # Then
        expect = np.cumsum(data)
        # print(result, y)
        self.assertTrue(np.all(expect[:-1] == result[1:]))
