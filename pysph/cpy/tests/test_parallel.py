from math import sin
import unittest
import numpy as np

from ..parallel import get_all_functions, Elementwise


def f(x):
    return sin(x) + 1


def g(x):
    return f(x*2)


class TestParallelUtils(unittest.TestCase):
    def test_get_all_functions(self):
        # Given/When
        result = get_all_functions(g)

        # Then
        expect = [f]
        self.assertEqual(expect, result)

    def test_elementwise_works_with_cython(self):
        # Given
        def axpb(i=0, x=[0.0], y=[0.0], a=1.0, b=1.0):
            y[i] = a*sin(x[i]) + b

        x = np.linspace(0, 1, 10000000)
        y = np.zeros_like(x)
        a = 2.0
        b = 3.0

        # When
        e = Elementwise(axpb, backend='cython')
        e(x, y, a, b)

        # Then
        self.assertTrue(np.allclose(y, a*np.sin(x) + b))
