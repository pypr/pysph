import numpy as np
from unittest import TestCase, main

from pysph.base.reduce_array import serial_reduce_array, dummy_reduce_array


class TestSerialReduceArray(TestCase):
    def test_reduce_sum_works(self):
        x = np.linspace(0, 10, 100)
        expect = np.sum(x)
        result = serial_reduce_array(x, 'sum')
        self.assertAlmostEqual(result, expect)

    def test_reduce_prod_works(self):
        x = np.linspace(0, 10, 100)
        expect = np.prod(x)
        result = serial_reduce_array(x, 'prod')
        self.assertAlmostEqual(result, expect)

    def test_reduce_max_works(self):
        x = np.linspace(0, 10, 100)
        expect = np.max(x)
        result = serial_reduce_array(x, 'max')
        self.assertAlmostEqual(result, expect)

    def test_reduce_min_works(self):
        x = np.linspace(0, 10, 100)
        expect = np.min(x)
        result = serial_reduce_array(x, 'min')
        self.assertAlmostEqual(result, expect)

    def test_reduce_raises_error_for_wrong_op(self):
        x = np.linspace(0, 10, 100)
        self.assertRaises(RuntimeError, serial_reduce_array, x, 'foo')

    def test_dummy_reduce_array_does_nothing(self):
        x = np.array([1.0, 2.0])
        expect = x
        result = dummy_reduce_array(x, 'min')
        self.assertTrue(np.alltrue(result == expect))


if __name__ == '__main__':
    main()
