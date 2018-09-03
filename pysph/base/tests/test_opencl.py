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

    def test_append_with_reallocation(self):
        # Given
        dev_array = self.make_dev_array()

        # When
        dev_array.append(2)

        # Then
        assert dev_array.array[-1] == 2
        assert len(dev_array.get_data()) == 32

    def test_append_without_reallocation(self):
        # Given
        dev_array = self.make_dev_array()
        dev_array.reserve(20)

        # When
        dev_array.append(2)

        # Then
        assert dev_array.array[-1] == 2
        assert len(dev_array.get_data()) == 20

    def test_extend(self):
        # Given
        dev_array = self.make_dev_array()
        cl_array = 2 + cl.array.zeros(get_queue(), 64, dtype=np.int32)

        # When
        dev_array.extend(cl_array)

        # Then
        assert np.all(dev_array.array[-len(cl_array)].get() == cl_array.get())

    def test_remove(self):
        # Given
        dev_array = DeviceArray(np.int32)
        orig_array = cl.array.arange(get_queue(), 0, 16, 1, dtype=np.int32)
        dev_array.set_data(orig_array)
        indices = cl.array.arange(get_queue(), 0, 8, 1, dtype=np.int32)

        # When
        dev_array.remove(indices)

        # Then
        assert np.all(dev_array.array.get() == (8 + indices).get())

    def test_align(self):
        # Given
        dev_array = DeviceArray(np.int32)
        orig_array = cl.array.arange(get_queue(), 0, 16, 1, dtype=np.int32)
        dev_array.set_data(orig_array)
        indices = cl.array.arange(get_queue(), 15, -1, -1, dtype=np.int32)

        # When
        dev_array.align(indices)

        # Then
        assert np.all(dev_array.array.get() == indices.get())

    def test_squeeze(self):
        # Given
        dev_array = self.make_dev_array()
        dev_array.fill(2)
        dev_array.reserve(32)
        assert dev_array.alloc == 32

        # When
        dev_array.squeeze()

        # Then
        assert dev_array.alloc == 16

    def test_copy_values(self):
        # Given
        dev_array = self.make_dev_array()
        dev_array.fill(2)

        dest = cl.array.empty(get_queue(), 8, dtype=np.int32)
        indices = cl.array.arange(get_queue(), 0, 8, 1, dtype=np.int32)

        # When
        dev_array.copy_values(indices, dest)

        # Then
        assert np.all(dev_array.array[:len(indices)].get() == dest.get())

    def test_min_max(self):
        # Given
        dev_array = self.make_dev_array()
        dev_array.fill(2)
        dev_array.array[0], dev_array.array[1] = 1, 10

        # When
        dev_array.update_min_max()

        # Then
        assert dev_array.minimum == 1
        assert dev_array.maximum == 10


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
        self.assertFalse('test' in h.properties)

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

    def test_get_number_of_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.align_particles()

        # Then
        assert h.get_number_of_particles() == 5
        assert h.get_number_of_particles(real=True) == 3

    def test_align(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))

        indices = cl.array.arange(get_queue(), 4, -1, -1, dtype=np.int32)

        h.align(indices)

        # Then
        assert np.all(h.x.get() == np.array([6., 5., 4., 3., 2.]))

    def test_align_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.align_particles()

        # Then
        x = h.x.get()
        assert np.all(np.sort(x[:-2]) == np.array([2., 3., 5.]))

    def test_remove_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.resize(4)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0], h.x.dtype))

        indices = np.array([1, 2], dtype=np.uint32)
        indices = cl.array.to_device(get_queue(), indices)

        h.remove_particles(indices)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([2., 5.]))

    def test_remove_tagged_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.remove_tagged_particles(1)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([2., 3., 5.]))

    def test_add_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)
        x = cl.array.zeros(get_queue(), 4, np.float32)

        h.add_particles(x=x)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([0., 0., 0., 0., 0., 1.]))

    def test_extend(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa)

        # When
        pa.set_device_helper(h)

        h.extend(4)

        # Then
        assert h.get_number_of_particles() == 6

    def test_append_parray(self):
        # Given
        pa1 = self.pa
        pa2 = get_particle_array(name='s', x=[0.0, 1.0], m=1.0, rho=2.0)
        h = DeviceHelper(pa1)
        pa1.set_device_helper(h)

        # When
        h.append_parray(pa2)

        # Then
        assert h.get_number_of_particles() == 4

    def test_extract_particles(self):
        # Given
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                m=1.0, rho=2.0)
        h = DeviceHelper(pa)
        pa.set_device_helper(h)

        # When
        indices = np.array([1, 2], dtype=np.uint32)
        indices = cl.array.to_device(get_queue(), indices)

        result_pa = h.extract_particles(indices)

        # Then
        assert result_pa.gpu.get_number_of_particles() == 2
