from unittest import TestCase
import pytest
import numpy as np

pytest.importorskip("pysph.base.opencl")

import pyopencl as cl

from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.opencl import DeviceHelper, Array, \
        get_queue  # noqa: E402
from pysph.cpy.config import get_config
from pysph.cpy.array import Array
import pysph.cpy.array as array


class DeviceHelperTest(object):
    def setup(self):
        self.pa = get_particle_array(name='f', x=[0.0, 1.0], m=1.0, rho=2.0)

    def test_simple(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)
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
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)
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
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

        # Then
        self.assertEqual(h.max('x'), 1.0)

    def test_that_adding_removing_prop_to_array_updates_gpu(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))

        indices = array.arange(4, -1, -1, dtype=np.int32,
                               backend=self.backend)

        h.align(indices)

        # Then
        assert np.all(h.x.get() == np.array([6., 5., 4., 3., 2.]))

    def test_align_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

        # When
        pa.set_device_helper(h)
        h.resize(4)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0], h.x.dtype))

        indices = np.array([1, 2], dtype=np.uint32)
        indices = array.to_device(indices, backend=self.backend)

        h.remove_particles(indices)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([2., 5.]))

    def test_remove_tagged_particles(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=self.backend)

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
        h = DeviceHelper(pa, backend=self.backend)

        # When
        pa.set_device_helper(h)
        x = array.zeros(4, np.float32, backend=self.backend)

        h.add_particles(x=x)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([0., 0., 0., 0., 0., 1.]))

    def test_extend(self):
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=self.backend)

        # When
        pa.set_device_helper(h)

        h.extend(4)

        # Then
        assert h.get_number_of_particles() == 6

    def test_append_parray(self):
        # Given
        pa1 = self.pa
        pa2 = get_particle_array(name='s', x=[0.0, 1.0], m=1.0, rho=2.0)
        h = DeviceHelper(pa1, backend=self.backend)
        pa1.set_device_helper(h)

        # When
        h.append_parray(pa2)

        # Then
        assert h.get_number_of_particles() == 4

    def test_extract_particles(self):
        # Given
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                m=1.0, rho=2.0)
        h = DeviceHelper(pa, backend=self.backend)
        pa.set_device_helper(h)

        # When
        indices = np.array([1, 2], dtype=np.uint32)
        indices = array.to_device(indices, backend=self.backend)

        result_pa = h.extract_particles(indices)

        # Then
        assert result_pa.gpu.get_number_of_particles() == 2

class DeviceHelperTestCython(DeviceHelperTest, TestCase):
    def setUp(self):
        self.setup()
        self.backend = 'cython'
        get_config().use_openmp = True

class DeviceHelperTestOpenCL(DeviceHelperTest, TestCase):
    def setUp(self):
        self.setup()
        self.backend = 'opencl'
        pytest.importorskip('pyopencl')

class DeviceHelperTestCUDA(DeviceHelperTest, TestCase):
    def setUp(self):
        self.setup()
        self.backend = 'cuda'
        pytest.importorskip('pycuda')

