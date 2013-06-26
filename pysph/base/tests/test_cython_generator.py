"""Test code for Cython code generation.
"""
import unittest
from textwrap import dedent

from pysph.base.cython_generator import (CythonGenerator, CythonClassHelper,
    all_numeric)

class BasicEq:
    def __init__(self, hidden=None, rho=0.0, c=0.0):
        self.rho = rho
        self.c = c
        self._hidden = ['a', 'b']

class EqWithMethod(BasicEq):
    def func(self, d_idx=0, d_x=[0.0, 0.0]):
        tmp = abs(self.rho*self.c)
        d_x[d_idx] = d_x[d_idx]*tmp


class TestBase(unittest.TestCase):
    def assert_code_equal(self, result, expect):
        expect = expect.strip()
        result = result.strip()
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(expect, result, msg)


class TestMiscUtils(TestBase):

    def test_all_numeric(self):
        x = [1, 2, 3.0]
        self.assertTrue(all_numeric(x))
        x = [0.0, 1, 3L]
        self.assertTrue(all_numeric(x))
        x = [0.0, 1.0, '']
        self.assertFalse(all_numeric(x))

    def test_detect_type(self):
        cases = [(('d_something', None), 'double*'),
                 (('s_something', None), 'double*'),
                 (('d_idx', 0), 'long'),
                 (('x', 1), 'long'),
                 (('s', 'asdas'), 'str'),
                 (('junk', 1.0), 'double'),
                 (('y', [0.0, 1.0]), 'double[2]'),
                 (('y', [0.0, 1.0, 0.0]), 'double[3]'),
                 (('y', None), 'object'),
                ]
        cg = CythonGenerator()
        for args, expect in cases:
            msg = 'detect_type(*%r) != %r'%(args, expect)
            self.assertEqual(cg.detect_type(*args), expect, msg)


    def test_cython_class_helper(self):
        code = ('def f(self, x):',
                '        x += 1\n        return x+1')
        c = CythonClassHelper(name='A', public_vars={'x': 'double'},
                              methods=[code])
        expect = dedent("""
        cdef class A:
            cdef public double x
            def __init__(self, object obj):
                for key in obj.__dict__:
                    setattr(self, key, getattr(obj, key))

            def f(self, x):
                x += 1
                return x+1
        """)
        self.assert_code_equal(c.generate().strip(), expect.strip())


class TestCythonCodeGenerator(TestBase):
    def test_simple_constructor(self):
        cg = CythonGenerator()
        cg.parse(BasicEq())
        expect = dedent("""
        cdef class BasicEq:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, object obj):
                for key in obj.__dict__:
                    setattr(self, key, getattr(obj, key))
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_simple_method(self):
        cg = CythonGenerator()
        cg.parse(EqWithMethod())
        expect = dedent("""
        cdef class EqWithMethod:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, object obj):
                for key in obj.__dict__:
                    setattr(self, key, getattr(obj, key))

            cdef inline func(self, long d_idx, double* d_x):
                cdef double tmp
                tmp = abs(self.rho*self.c)
                d_x[d_idx] = d_x[d_idx]*tmp
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())


if __name__ == '__main__':
    unittest.main()
