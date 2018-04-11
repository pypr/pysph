import unittest

import numpy as np

from ..types import KnownType, declare, annotate


class TestDeclare(unittest.TestCase):
    def test_declare(self):
        self.assertEqual(declare('int'), 0)
        self.assertEqual(declare('long'), 0)
        self.assertEqual(declare('double'), 0.0)
        self.assertEqual(declare('float'), 0.0)

        self.assertEqual(declare('int', 2), (0, 0))
        self.assertEqual(declare('long', 3), (0, 0, 0))
        self.assertEqual(declare('double', 2), (0.0, 0.0))
        self.assertEqual(declare('float', 3), (0.0, 0.0, 0.0))

        res = declare('matrix(3)')
        self.assertTrue(np.all(res == np.zeros(3)))
        res = declare('matrix(3)', 3)
        for i in range(3):
            self.assertTrue(np.all(res[0] == np.zeros(3)))
        res = declare('matrix((3,))')
        self.assertTrue(np.all(res == np.zeros(3)))
        res = declare('matrix((3, 3))')
        self.assertTrue(np.all(res == np.zeros((3, 3))))

    def test_declare_with_type(self):
        res = declare('matrix(3, "int")')
        self.assertTrue(np.all(res == np.zeros(3)))
        self.assertEqual(res.dtype, np.int32)

        res = declare('matrix((2, 2), "unsigned int")')
        self.assertTrue(np.all(res == np.zeros((2, 2))))
        self.assertEqual(res.dtype, np.uint32)

        res = declare('matrix((3,), "float")')
        self.assertTrue(np.all(res == np.zeros((3,))))
        self.assertEqual(res.dtype, np.float32)

    def test_declare_with_address_space(self):
        self.assertEqual(declare('LOCAL_MEM int', 2), (0, 0))
        self.assertEqual(declare('GLOBAL_MEM float', 2), (0.0, 0.0))

        res = declare('LOCAL_MEM matrix(3)')
        self.assertTrue(np.all(res == np.zeros(3)))

        res = declare('GLOBAL_MEM matrix(3)')
        self.assertTrue(np.all(res == np.zeros(3)))


class TestAnnotate(unittest.TestCase):
    def test_simple_annotation(self):
        # Given/When
        @annotate(i='int', x='floatp', return_='float')
        def f(i, x):
            return x[i]*2.0

        # Then
        result = f.__annotations__
        self.assertEqual(result['return'], KnownType('float'))
        self.assertEqual(result['i'], KnownType('int'))
        self.assertEqual(result['x'], KnownType('float*', 'float'))

    def test_reversed_annotation(self):
        # Given/When
        @annotate(i='int', floatp='x, y', return_='float')
        def f(i, x, y):
            return x[i]*y[i]

        # Then
        result = f.__annotations__
        self.assertEqual(result['return'], KnownType('float'))
        self.assertEqual(result['i'], KnownType('int'))
        self.assertEqual(result['x'], KnownType('float*', 'float'))
        self.assertEqual(result['y'], KnownType('float*', 'float'))

    def test_decorator_accepts_known_type_instance(self):
        # Given/When
        @annotate(x=KnownType('Thing'))
        def f(x):
            x.f()

        # Then
        result = f.__annotations__
        self.assertEqual(result['x'], KnownType('Thing'))

    def test_decorator_raises_error_for_unknown_error(self):
        def f(x):
            pass

        self.assertRaises(TypeError, annotate, f, x='alpha')
