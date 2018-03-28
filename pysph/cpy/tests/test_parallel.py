from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..array import wrap
from ..parallel import Elementwise


class TestParallelUtils(unittest.TestCase):

    def _check_simple_elementwise(self, backend):
        # Given
        def axpb(i=0, x=[0.0], y=[0.0], a=1.0, b=1.0):
            y[i] = a*sin(x[i]) + b

        x = np.linspace(0, 1, 10000000)
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
