from unittest import TestCase
import pytest
import numpy as np

pytest.importorskip("pysph.base.opencl")

import pyopencl as cl

from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.opencl import DeviceHelper, DeviceArray  # noqa: E402


class TestDeviceArray(TestCase):
    def setUp(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.dev_array = DeviceArray(self.queue, np.int32)

    def tearDown(self):
        self.dev_array = DeviceArray(self.queue, np.int32)

    def test_reserve(self):
        self.dev_array.reserve(64)
        assert len(self.dev_array.get_array()) == 64
        assert self.dev_array.length == 0

    def test_resize_with_reallocation(self):
        self.dev_array.resize(64)
        assert len(self.dev_array.get_array()) == 64
        assert self.dev_array.length == 64

    def test_resize_without_reallocation(self):
        self.dev_array = DeviceArray(self.queue, np.int32, n=128)

        self.dev_array.resize(64)
        assert len(self.dev_array.get_array()) == 128
        assert self.dev_array.length == 64

    def test_copy(self):
        self.dev_array = DeviceArray(self.queue, np.int32, n=16)
        self.dev_array.fill(0)
        dev_array_copy = self.dev_array.copy()
        assert np.all(self.dev_array.data.get() == dev_array_copy.data.get())

        dev_array_copy.data[0] = 1

        assert self.dev_array.data[0] != dev_array_copy.data[0]


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
