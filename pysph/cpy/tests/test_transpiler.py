from math import sin
import unittest

from ..transpiler import get_all_functions, Transpiler


def h(x=0.0):
    return sin(x) + 1


def f(x=0.0):
    return h(x*2+1)


def g(x=0.0):
    return f(x*2)


class TestTranspiler(unittest.TestCase):
    def test_get_all_functions(self):
        # Given/When
        result = get_all_functions(g)

        # Then
        expect = [f]
        self.assertEqual(expect, result)

    def test_transpiler(self):
        # Given
        t = Transpiler(backend='cython')

        # When
        t.add(g)

        # Then
        for func in (g, f, h):
            self.assertTrue(func in t.blocks)

        expect = [h, f, g]
        self.assertListEqual([x.obj for x in t.blocks], expect)
