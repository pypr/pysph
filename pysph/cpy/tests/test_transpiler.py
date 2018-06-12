from math import sin
import unittest

from ..transpiler import get_external_symbols_and_calls, Transpiler
from ..extern import printf

SIZE = 10

my_printf = printf


def h(x=0.0):
    return sin(x) + 1


def f(x=0.0):
    return h(x*2+1)


def g(x=0.0):
    return f(x*2)


def implicit_f(x, y):
    # These should be ignored.
    j = LID_0 + GID_0 + LDIM_0 + GDIM_0
    s = y[SIZE-1]
    for i in range(SIZE):
        s += sin(x[i])

    my_printf("%f", s)
    return s


def undefined_call(x):
    # An intentional error that should be caught.
    foo(x)


class TestTranspiler(unittest.TestCase):
    def test_get_external_symbols_and_calls(self):
        # Given/When
        syms, implicit, calls, ext = get_external_symbols_and_calls(
            g, 'cython'
        )

        # Then
        expect = [f]
        self.assertEqual(syms, {})
        self.assertEqual(expect, calls)
        self.assertEqual(ext, [])

        # Given/When
        syms, implicit, calls, ext = get_external_symbols_and_calls(
            implicit_f, 'cython'
        )

        # Then
        self.assertEqual(syms, {'SIZE': 10})
        self.assertEqual(implicit, {'i'})
        self.assertEqual(calls, [])
        self.assertEqual(ext, [my_printf])

        # Given/When
        self.assertRaises(NameError, get_external_symbols_and_calls,
                          undefined_call, 'cython')

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
