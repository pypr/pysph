from unittest import TestCase
import pytest
import numpy as np

pytest.importorskip("pysph.base.opencl")

import pyopencl as cl

from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.opencl import DeviceHelper, DeviceArray, \
        get_queue  # noqa: E402


class TestDeviceArray(TestCase):
    def make_dev_array(self, n=16):
        dev_array = DeviceArray(np.int32, n=n)
        dev_array.fill(0)
        dev_array.array[0] = 1
        return dev_array

    def test_reserve(self):
        # Given
        dev_array = self.make_dev_array()

        # When
        dev_array.reserve(64)

        # Then
        assert len(dev_array.get_data()) == 64
        assert dev_array.length == 16
        assert dev_array.array[0] == 1

    def test_resize_with_reallocation(self):
        # Given
        dev_array = self.make_dev_array()

        # When
        dev_array.resize(64)

        # Then
        assert len(dev_array.get_data()) == 64
        assert dev_array.length == 64
        assert dev_array.array[0] == 1

    def test_resize_without_reallocation(self):
        # Given
        dev_array = self.make_dev_array(n=128)

        # When
        dev_array.resize(64)

        # Then
        assert len(dev_array.get_data()) == 128
        assert dev_array.length == 64
        assert dev_array.array[0] == 1

    def test_copy(self):
        # Given
        dev_array = self.make_dev_array()

        # When
        dev_array_copy = dev_array.copy()

        # Then
        assert np.all(dev_array.array.get() == dev_array_copy.array.get())

        dev_array_copy.array[0] = 2
        assert dev_array.array[0] != dev_array_copy.array[0]


class TestDeviceHelper(TestCase):
    def setUp(self):
        self.pa = get_particle_array(name='f', x=[0.0, 1.0], m=1.0, rho=2.0)

    def test_simple(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)

        # Then
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))

    def test_push_correctly_sets_values_with_args(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)
        self.assertEqual(pa.tag[0], 0)

        # When
        pa.set_device_helper(h)
        pa.x[:] = [2.0, 3.0]
        pa.rho[0] = 1.0
        pa.tag[:] = 1
        h.push('x', 'rho', 'tag')

        # Then
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))

    def test_push_correctly_sets_values_with_no_args(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        pa.x[:] = 1.0
        pa.rho[:] = 1.0
        pa.m[:] = 1.0
        pa.tag[:] = [1, 2]
        h.push()

        # Then
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))

    def test_pull_correctly_sets_values_with_args(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)
        self.assertEqual(pa.tag[0], 0)

        # When
        pa.set_device_helper(h)
        h.x.set(np.array([2.0, 3.0], h.x.dtype))
        h.rho[0] = 1.0
        h.tag[:] = 1
        h.pull('x', 'rho', 'tag')

        # Then
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))

    def test_pull_correctly_sets_values_with_no_args(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.x[:] = 1.0
        h.rho[:] = 1.0
        h.m[:] = 1.0
        h.tag[:] = np.array([1, 2], h.tag.dtype)
        h.pull()

        # Then
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))

    def test_max_provides_maximum(self):
        # Given/When
        pa = self.pa
        h = DeviceHelper(pa)

        # Then
        self.assertEqual(h.max('x'), 1.0)

    def test_that_adding_removing_prop_to_array_updates_gpu(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        pa.add_property('test', data=[3.0, 4.0])

        # Then
        self.assertTrue(np.allclose(pa.test, h.test.get()))

        # When
        pa.remove_property('test')

        # Then
        self.assertFalse(hasattr(h, 'test'))
        self.assertFalse('test' in h._data)
        self.assertFalse('test' in h._props)

    def test_resize_works(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        pa.extend(2)
        pa.align_particles()
        h.resize(4)

        # Then
        self.assertTrue(np.allclose(pa.x[:2], h.x[:2].get()))
        self.assertTrue(np.allclose(pa.m[:2], h.m[:2].get()))
        self.assertTrue(np.allclose(pa.rho[:2], h.rho[:2].get()))
        self.assertTrue(np.allclose(pa.tag[:2], h.tag[:2].get()))

        # When
        pa.remove_particles([2, 3])
        pa.align_particles()
        old_x = h.x.data
        h.resize(2)

        # Then
        self.assertEqual(old_x, h.x.data)
        self.assertTrue(np.allclose(pa.x, h.x.get()))
        self.assertTrue(np.allclose(pa.y, h.y.get()))
        self.assertTrue(np.allclose(pa.m, h.m.get()))
        self.assertTrue(np.allclose(pa.rho, h.rho.get()))
        self.assertTrue(np.allclose(pa.tag, h.tag.get()))
