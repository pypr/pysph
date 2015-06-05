"""
Tests for the carray module.

Only the LongArray is tested. As the code in carray.pyx is auto-generated, tests
for one class hould suffice.
"""


# standard imports
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy

# local imports
from pyzoltan.core.carray import LongArray, py_aligned


class TestAligned(unittest.TestCase):

    def test_aligned_to_64_bits(self):
        self.assertEqual(py_aligned(12, 1), 64)
        self.assertEqual(py_aligned(1, 1), 64)
        self.assertEqual(py_aligned(64, 1), 64)
        self.assertEqual(py_aligned(120, 1), 128)

        self.assertEqual(py_aligned(1, 2), 32)
        self.assertEqual(py_aligned(12, 2), 32)
        self.assertEqual(py_aligned(32, 2), 32)
        self.assertEqual(py_aligned(33, 2), 64)

        self.assertEqual(py_aligned(1, 3), 64)
        self.assertEqual(py_aligned(65, 3), 256)

        self.assertEqual(py_aligned(1, 4), 16)
        self.assertEqual(py_aligned(16, 4), 16)
        self.assertEqual(py_aligned(21, 4), 32)

        self.assertEqual(py_aligned(1, 5), 64)
        self.assertEqual(py_aligned(13, 5), 128)

        self.assertEqual(py_aligned(1, 8), 8)
        self.assertEqual(py_aligned(8, 8), 8)
        self.assertEqual(py_aligned(11, 8), 16)


class TestLongArray(unittest.TestCase):
    """
    Tests for the LongArray class.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """
        l = LongArray(10)

        self.assertEqual(l.length, 10)
        self.assertEqual(l.alloc, 10)
        self.assertEqual(len(l.get_npy_array()), 10)

        l = LongArray()

        self.assertEqual(l.length, 0)
        self.assertEqual(l.alloc, 16)
        self.assertEqual(len(l.get_npy_array()), 0)

    def test_get_set_indexing(self):
        """
        Test get/set and [] operator.
        """
        l = LongArray(10)
        l.set(0, 10)
        l.set(9, 1)

        self.assertEqual(l.get(0), 10)
        self.assertEqual(l.get(9), 1)

        l[9] = 2
        self.assertEqual(l[9], 2)

    def test_append(self):
        """
        Test the append function.
        """
        l = LongArray(0)
        l.append(1)
        l.append(2)
        l.append(3)

        self.assertEqual(l.length, 3)
        self.assertEqual(l[0], 1)
        self.assertEqual(l[1], 2)
        self.assertEqual(l[2], 3)

    def test_reserve(self):
        """
        Tests the reserve function.
        """
        l = LongArray(0)
        l.reserve(10)

        self.assertEqual(l.alloc, 16)
        self.assertEqual(l.length, 0)
        self.assertEqual(len(l.get_npy_array()), 0)

        l.reserve(20)
        self.assertEqual(l.alloc, 20)
        self.assertEqual(l.length,  0)
        self.assertEqual(len(l.get_npy_array()), 0)

    def test_resize(self):
        """
        Tests the resize function.
        """
        l = LongArray(0)

        l.resize(20)
        self.assertEqual(l.length, 20)
        self.assertEqual(len(l.get_npy_array()), 20)
        self.assertEqual(l.alloc >= l.length, True)

    def test_get_npy_array(self):
        """
        Tests the get_npy_array array.
        """
        l = LongArray(3)
        l[0] = 1
        l[1] = 2
        l[2] = 3

        nparray = l.get_npy_array()
        self.assertEqual(len(nparray), 3)

        for i in range(3):
            self.assertEqual(nparray[0], l[0])

    def test_set_data(self):
        """
        Tests the set_data function.
        """
        l = LongArray(5)
        np = numpy.arange(5)
        l.set_data(np)

        for i in range(5):
            self.assertEqual(l[i], np[i])

        self.assertRaises(ValueError, l.set_data, numpy.arange(10))

    def test_squeeze(self):
        """
        Tests the squeeze function.
        """
        l = LongArray(5)
        l.append(4)

        self.assertEqual(l.alloc > l.length, True)

        l.squeeze()

        self.assertEqual(l.length, 6)
        self.assertEqual(l.alloc >= l.length, True)
        self.assertEqual(len(l.get_npy_array()), 6)

    def test_squeeze_for_zero_length_array(self):
        # Given.
        l = LongArray()

        # When
        l.squeeze()

        # Then
        self.assertEqual(l.length, 0)
        self.assertEqual(len(l.get_npy_array()), 0)
        self.assertEqual(l.alloc >= l.length, True)
        del l # This should work and not segfault.

    def test_reset(self):
        """
        Tests the reset function.
        """
        l = LongArray(5)
        l.reset()

        self.assertEqual(l.length, 0)
        self.assertEqual(l.alloc, 5)
        self.assertEqual(len(l.get_npy_array()), 0)

    def test_extend(self):
        """
        Tests the extend function.
        """
        l1 = LongArray(5)

        for i in range(5):
            l1[i] = i

        l2 = LongArray(5)

        for i in range(5):
            l2[i] = 5 + i

        l1.extend(l2.get_npy_array())

        self.assertEqual(l1.length, 10)
        self.assertEqual(numpy.allclose(l1.get_npy_array(), numpy.arange(10)), True)

    def test_remove(self):
        """
        Tests the remove function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))
        rem = [0, 4, 3]
        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 7)
        self.assertEqual(numpy.allclose([7, 1, 2, 8, 9, 5, 6],
                                        l1.get_npy_array()), True)

        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 4)
        self.assertEqual(numpy.allclose([6, 1, 2, 5], l1.get_npy_array()), True)

        rem = [0, 1, 3]
        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 1)
        self.assertEqual(numpy.allclose([2], l1.get_npy_array()), True)

        l1.remove(numpy.array([0], dtype=numpy.int))
        self.assertEqual(l1.length, 0)
        self.assertEqual(len(l1.get_npy_array()), 0)

    def test_align_array(self):
        """
        Test the align_array function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        new_indices = LongArray(10)
        new_indices.set_data(numpy.asarray([1, 5, 3, 2, 4, 7, 8, 6, 9, 0]))

        l1.align_array(new_indices)
        self.assertEqual(numpy.allclose([1, 5, 3, 2, 4, 7, 8, 6, 9, 0],
                                        l1.get_npy_array()), True)

    def test_copy_subset(self):
        """
        Tests the copy_subset function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        l2 = LongArray(4)
        l2[0] = 4
        l2[1] = 3
        l2[2] = 2
        l2[3] = 1

        # a valid copy.
        l1.copy_subset(l2, 5, 9)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 4, 3, 2, 1, 9],
                                        l1.get_npy_array()), True)

        # try to copy different sized arrays without any index specification.
        l1.set_data(numpy.arange(10))
        # copy to the last k values of source array.
        l1.copy_subset(l2, start_index=6)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
                                        l1.get_npy_array()), True)

        l1.set_data(numpy.arange(10))
        l1.copy_subset(l2, start_index=7)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 5, 6, 4, 3, 2],
                                        l1.get_npy_array()), True)

        # some invalid operations.
        l1.set_data(numpy.arange(10))
        self.assertRaises(ValueError, l1.copy_subset, l2, -1, 1)
        self.assertRaises(ValueError, l1.copy_subset, l2, 3, 2)
        self.assertRaises(ValueError, l1.copy_subset, l2, 0, 11)
        self.assertRaises(ValueError, l1.copy_subset, l2, 10, 20)
        self.assertRaises(ValueError, l1.copy_subset, l2, -1, -1)

    def test_update_min_max(self):
        """
        Tests the update_min_max function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        l1.update_min_max()

        self.assertEqual(l1.minimum, 0)
        self.assertEqual(l1.maximum, 9)

        l1[9] = -1
        l1[0] = -20
        l1[4] = 200
        l1.update_min_max()

        self.assertEqual(l1.minimum, -20)
        self.assertEqual(l1.maximum, 200)

    def test_pickling(self):
        """
        Tests the __reduce__ and __setstate__ functions.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        import pickle

        l1_dump = pickle.dumps(l1)

        l1_load = pickle.loads(l1_dump)
        self.assertEqual((l1_load.get_npy_array() == l1.get_npy_array()).all(), True)

    def test_set_view(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.
        view = LongArray()
        view.set_view(src, 1, 4)

        # Then.
        self.assertEqual(view.length, 3)
        expect = list(range(1, 4))
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_set_view_for_empty_array(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.
        view = LongArray()
        view.set_view(src, 1, 1)

        # Then.
        self.assertEqual(view.length, 0)
        expect = []
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_set_view_stores_reference_to_parent(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When
        view = LongArray()
        view.set_view(src, 1, 4)
        del src

        # Then.
        self.assertEqual(view.length, 3)
        expect = list(range(1, 4))
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_reset_works_after_set_view(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))
        view = LongArray()
        view.set_view(src, 1, 3)

        # When.
        view.reset()
        view.extend(numpy.arange(3)*10)

        # Then.
        self.assertEqual(view.length, 3)
        expect = (numpy.arange(3)*10).tolist()
        self.assertListEqual(view.get_npy_array().tolist(), expect)



if __name__ == '__main__':
    unittest.main()
