"""
Tests for the particle array module.
"""

# standard imports
import unittest
import numpy

# local imports
from pysph.base import particle_array
from pysph.base import utils

from cyarray.carray import LongArray, IntArray, DoubleArray

import pickle
import pytest

from compyle.config import get_config


def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)


###############################################################################
# `ParticleArrayTest` class.
###############################################################################
class ParticleArrayTest(object):
    """
    Tests for the particle array class.
    """

    def test_constructor(self):
        # Default constructor test.
        p = particle_array.ParticleArray(name='test_particle_array',
                                         backend=self.backend)

        self.assertEqual(p.name, 'test_particle_array')
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual(p.properties['tag'].length, 0)

        # Constructor with some properties.
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        p = particle_array.ParticleArray(x={'data': x},
                                         y={'data': y},
                                         z={'data': z},
                                         m={'data': m},
                                         h={'data': h},
                                         backend=self.backend)

        self.assertEqual(p.name, '')

        self.assertEqual('x' in p.properties, True)
        self.assertEqual('y' in p.properties, True)
        self.assertEqual('z' in p.properties, True)
        self.assertEqual('m' in p.properties, True)
        self.assertEqual('h' in p.properties, True)

        # get the properties are check if they are the same
        xarr = p.properties['x'].get_npy_array()
        self.assertEqual(check_array(xarr, x), True)

        yarr = p.properties['y'].get_npy_array()
        self.assertEqual(check_array(yarr, y), True)

        zarr = p.properties['z'].get_npy_array()
        self.assertEqual(check_array(zarr, z), True)

        marr = p.properties['m'].get_npy_array()
        self.assertEqual(check_array(marr, m), True)

        harr = p.properties['h'].get_npy_array()
        self.assertEqual(check_array(harr, h), True)

        # check if the 'tag' array was added.
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual(list(p.properties.values())[0].length == len(x), True)

        # Constructor with tags
        tags = [0, 1, 0, 1]
        p = particle_array.ParticleArray(x={'data': x}, y={'data': y},
                                         z={'data': z},
                                         tag={'data': tags, 'type': 'int'},
                                         backend=self.backend)
        self.assertEqual(check_array(p.get('tag', only_real_particles=False),
                                     [0, 0, 1, 1]), True)
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [1, 3, 2, 4]), True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [0, 2, 1, 3]), True)
        self.assertEqual(check_array(p.get('z', only_real_particles=False),
                                     [0, 0, 0, 0]), True)

        # trying to create particle array without any values but some
        # properties.
        p = particle_array.ParticleArray(x={}, y={}, z={}, h={},
                                         backend=self.backend)

        self.assertEqual(p.get_number_of_particles(), 0)
        self.assertEqual('x' in p.properties, True)
        self.assertEqual('y' in p.properties, True)
        self.assertEqual('z' in p.properties, True)
        self.assertEqual('tag' in p.properties, True)

        # now trying to supply some properties with values and others without
        p = particle_array.ParticleArray(
            x={'default': 10.0}, y={'data': [1.0, 2.0]},
            z={}, h={'data': [0.1, 0.1]}, backend=self.backend
        )

        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [10., 10.]), True)
        self.assertEqual(check_array(p.y, [1., 2.]), True)
        self.assertEqual(check_array(p.z, [0, 0]), True)
        self.assertEqual(check_array(p.h, [0.1, 0.1]), True)

    def test_constructor_works_with_strides(self):
        # Given
        x = [1, 2, 3, 4.]
        rho = 10.0
        data = numpy.arange(8)

        # When
        p = particle_array.ParticleArray(
            x=x, rho=rho, data={'data': data, 'stride': 2}, name='fluid',
            backend=self.backend
        )

        # Then
        self.assertEqual(p.name, 'fluid')

        self.assertEqual('x' in p.properties, True)
        self.assertEqual('rho' in p.properties, True)
        self.assertEqual('data' in p.properties, True)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual('pid' in p.properties, True)
        self.assertEqual('gid' in p.properties, True)

        # get the properties are check if they are the same
        self.assertEqual(check_array(p.x, x), True)
        self.assertEqual(check_array(p.rho, numpy.ones(4) * rho), True)
        self.assertEqual(check_array(p.data, numpy.ravel(data)), True)

    def test_constructor_works_with_simple_props(self):
        # Given
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        rho = 10.0
        data = numpy.diag((2, 2))

        # When
        p = particle_array.ParticleArray(
            x=x, y=y, rho=rho, data=data, name='fluid',
            backend=self.backend
        )

        # Then
        self.assertEqual(p.name, 'fluid')

        self.assertEqual('x' in p.properties, True)
        self.assertEqual('y' in p.properties, True)
        self.assertEqual('rho' in p.properties, True)
        self.assertEqual('data' in p.properties, True)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual('pid' in p.properties, True)
        self.assertEqual('gid' in p.properties, True)

        # get the properties are check if they are the same
        self.assertEqual(check_array(p.x, x), True)
        self.assertEqual(check_array(p.y, y), True)
        self.assertEqual(check_array(p.rho, numpy.ones(4) * rho), True)
        self.assertEqual(check_array(p.data, numpy.ravel(data)), True)

    def test_get_number_of_particles(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        self.assertEqual(p.get_number_of_particles(), 4)

    def test_get(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        self.assertEqual(check_array(x, p.get('x')), True)
        self.assertEqual(check_array(y, p.get('y')), True)
        self.assertEqual(check_array(z, p.get('z')), True)
        self.assertEqual(check_array(m, p.get('m')), True)
        self.assertEqual(check_array(h, p.get('h')), True)
        self.assertEqual(check_array(A.ravel(), p.get('A')), True)

    def test_clear(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data': x}, y={'data': y},
                                         z={'data': z}, m={'data': m},
                                         h={'data': h}, backend=self.backend)

        p.clear()

        self.assertEqual(len(p.properties), 3)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual(p.properties['tag'].length, 0)

        self.assertEqual('pid' in p.properties, True)
        self.assertEqual(p.properties['pid'].length, 0)

        self.assertEqual('gid' in p.properties, True)
        self.assertEqual(p.properties['gid'].length, 0)

    def test_getattr(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        self.assertEqual(check_array(x, p.x), True)
        self.assertEqual(check_array(y, p.y), True)
        self.assertEqual(check_array(z, p.z), True)
        self.assertEqual(check_array(m, p.m), True)
        self.assertEqual(check_array(h, p.h), True)
        self.assertEqual(check_array(A.ravel(), p.get('A')), True)

        # try getting an non-existant attribute
        self.assertRaises(AttributeError, p.__getattr__, 'a')

    def test_setattr(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        p.x = p.x * 2.0

        self.assertEqual(check_array(p.get('x'), [2., 4, 6, 8]), True)
        p.x = p.x + 3.0 * p.x
        self.assertEqual(check_array(p.get('x'), [8., 16., 24., 32.]), True)
        p.A = p.A*2
        self.assertEqual(check_array(p.get('A').ravel(), A*2), True)

    def test_remove_particles(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        remove_arr = LongArray(0)
        remove_arr.append(0)
        remove_arr.append(1)

        p.remove_particles(remove_arr)
        self.pull(p)

        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)
        self.assertEqual(check_array(p.A, numpy.arange(6, 12)), True)

        # now try invalid operations to make sure errors are raised.
        remove_arr.resize(10)
        self.assertRaises(ValueError, p.remove_particles, remove_arr)

        # now try to remove a particle with index more that particle
        # length.
        remove_arr = [2]

        p.remove_particles(remove_arr)
        self.pull(p)
        # make sure no change occurred.
        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)
        self.assertEqual(check_array(p.A, numpy.arange(6, 12)), True)

    def test_add_particles(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, A=dict(data=A, stride=3),
            backend=self.backend
        )

        new_particles = {}
        new_particles['x'] = numpy.array([5., 6, 7], dtype=numpy.float32)
        new_particles['y'] = numpy.array([4., 5, 6], dtype=numpy.float32)
        new_particles['z'] = numpy.array([0., 0, 0], dtype=numpy.float32)

        p.add_particles(**new_particles)
        self.pull(p)

        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        expect = numpy.zeros(21, dtype=A.dtype)
        expect[:12] = A
        numpy.testing.assert_array_equal(p.A, expect)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)

        # try adding an empty particle list
        p.add_particles(**{})
        self.pull(p)
        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(check_array(p.A, expect), True)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)

        # adding particles with tags
        p = particle_array.ParticleArray(x={'data': x}, y={'data': y},
                                         z={'data': z}, m={'data': m},
                                         h={'data': h},
                                         backend=self.backend)
        p.add_particles(x=[5, 6, 7, 8], tag=[1, 1, 0, 0])
        self.pull(p)

        self.assertEqual(p.get_number_of_particles(), 8)
        self.assertEqual(check_array(p.x, [1, 2, 3, 4, 7, 8]), True)
        self.assertEqual(check_array(p.y, [0, 1, 2, 3, 0, 0]), True)
        self.assertEqual(check_array(p.z, [0, 0, 0, 0, 0, 0]), True)

    def test_remove_tagged_particles(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        A = numpy.arange(12)
        tag = [1, 1, 1, 0]

        p = particle_array.ParticleArray(
            x={'data': x}, y={'data': y},
            z={'data': z}, m={'data': m},
            h={'data': h}, tag={'data': tag}, A={'data': A, 'stride': 3},
            backend=self.backend
        )

        numpy.testing.assert_array_equal(
            p.get('A'), numpy.arange(9, 12)
        )

        p.remove_tagged_particles(0)
        self.pull(p)

        self.assertEqual(p.get_number_of_particles(), 3)
        self.assertEqual(
            check_array(
                numpy.sort(p.get('x', only_real_particles=False)),
                [1., 2., 3.]),
            True
        )
        self.assertEqual(
            check_array(
                numpy.sort(p.get('y', only_real_particles=False)),
                [0., 1., 2.]
            ), True
        )
        self.assertEqual(
            check_array(
                numpy.sort(p.get('z', only_real_particles=False)),
                [0., 0, 0]
            ), True
        )
        self.assertEqual(
            check_array(
                numpy.sort(p.get('h', only_real_particles=False)),
                [0.1, 0.1, 0.1]
            ), True
        )
        self.assertEqual(
            check_array(p.get('m', only_real_particles=False), [1., 1., 1.]),
            True
        )
        if p.gpu is None:
            numpy.testing.assert_array_equal(
                p.get('A', only_real_particles=False),
                numpy.arange(9)
            )
        else:
            numpy.testing.assert_array_equal(
                p.get('A', only_real_particles=False),
                list(range(3, 9)) + [0., 1, 2]
            )
        self.assertEqual(check_array(p.x, []), True)
        self.assertEqual(check_array(p.y, []), True)
        self.assertEqual(check_array(p.z, []), True)
        self.assertEqual(check_array(p.h, []), True)
        self.assertEqual(check_array(p.m, []), True)
        self.assertEqual(check_array(p.A, []), True)

    def test_add_property(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        tag = [0, 0, 0, 0]

        p = particle_array.ParticleArray(x={'data': x}, y={'data': y},
                                         z={'data': z}, m={'data': m},
                                         h={'data': h}, tag={'data': tag},
                                         backend=self.backend)

        p.add_property(**{'name': 'x'})
        # make sure the current 'x' property is intact.
        self.assertEqual(check_array(p.x, x), True)

        # add a property with complete specification
        p.add_property(**{'name': 'f1',
                          'data': [1, 1, 2, 3],
                          'type': 'int',
                          'default': 4})
        self.assertEqual(check_array(p.f1, [1, 1, 2, 3]), True)
        self.assertEqual(type(p.properties['f1']), IntArray)
        self.assertEqual(p.default_values['f1'], 4)

        # add a property with stride.
        data = [1, 1, 2, 2, 3, 3, 4, 4]
        p.add_property(**{'name': 'm1',
                          'data': data,
                          'type': 'int',
                          'stride': 2})
        self.assertEqual(check_array(p.m1, data), True)
        self.assertEqual(type(p.properties['m1']), IntArray)

        # add a property without specifying the type
        p.add_property(**{'name': 'f2',
                          'data': [1, 1, 2, 3],
                          'default': 4.0})
        self.assertEqual(type(p.properties['f2']), DoubleArray)
        self.assertEqual(check_array(p.f2, [1, 1, 2, 3]), True)

        p.add_property(**{'name': 'f3'})
        self.assertEqual(type(p.properties['f3']), DoubleArray)
        self.assertEqual(p.properties['f3'].length,
                         p.get_number_of_particles())
        self.assertEqual(check_array(p.f3, [0, 0, 0, 0]), True)

        p.add_property(**{'name': 'f4', 'default': 3.0})
        self.assertEqual(type(p.properties['f4']), DoubleArray)
        self.assertEqual(p.properties['f4'].length,
                         p.get_number_of_particles())
        self.assertEqual(check_array(p.f4, [3, 3, 3, 3]), True)

        p.add_property('f5', data=10.0)
        self.assertEqual(type(p.properties['f5']), DoubleArray)
        self.assertEqual(p.properties['f5'].length,
                         p.get_number_of_particles())
        self.assertEqual(check_array(p.f5, [10.0, 10.0, 10.0, 10.0]), True)

        p.add_property('m2', data=10.0, stride=2)
        self.assertEqual(type(p.properties['m2']), DoubleArray)
        self.assertEqual(p.properties['m2'].length,
                         p.get_number_of_particles()*2)
        self.assertEqual(check_array(p.m2, [10.0]*8), True)

    def test_extend(self):
        # Given
        p = particle_array.ParticleArray(default_particle_tag=10, x={},
                                         y={'default': -1.},
                                         backend=self.backend)
        p.add_property('A', default=5.0, stride=2)

        # When
        p.extend(5)
        p.align_particles()
        self.pull(p)

        # Then
        self.assertEqual(p.get_number_of_particles(), 5)
        self.assertEqual(check_array(p.get(
            'x', only_real_particles=False), [0, 0, 0, 0, 0]), True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [-1., -1., -1., -1., -1.]), True)
        self.assertEqual(check_array(p.get('tag', only_real_particles=False),
                                     [10, 10, 10, 10, 10]), True)
        self.assertEqual(check_array(p.get('A', only_real_particles=False),
                                     [5.0]*10), True)

        # Given
        p = particle_array.ParticleArray(
            A={'data': [10.0, 10.0], 'stride': 2, 'default': -1.},
            backend=self.backend
        )

        # When
        p.extend(5)
        p.align_particles()
        self.pull(p)

        # Then
        self.assertEqual(check_array(p.get('A', only_real_particles=False),
                                     [10.0, 10.0] + [-1.0]*10), True)

    def test_resize(self):
        # Given
        p = particle_array.ParticleArray(
            A={'data': [10.0, 10.0], 'stride': 2, 'default': -1.},
            x=[1.0], backend=self.backend
        )

        # When
        p.resize(4)
        p.align_particles()
        self.pull(p)

        # Then
        self.assertEqual(p.get_carray('x').length, 4)
        self.assertEqual(p.get_carray('A').length, 8)

    def test_align_particles(self):
        # Given
        p = particle_array.ParticleArray(backend=self.backend)
        p.add_property(**{'name': 'x',
                          'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        p.add_property(**{'name': 'y',
                          'data': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
        a = numpy.arange(10) + 1
        A = numpy.zeros(20)
        A[::2] = a
        A[1::2] = a
        p.add_property('A', data=A, stride=2)
        print(A)
        p.set(**{'tag': [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]})

        # When
        self.push(p)
        p.align_particles()
        self.pull(p)

        # Then
        x_new = p.get('x', only_real_particles=False)
        y_new = p.get('y', only_real_particles=False)
        A_new = p.get('A', only_real_particles=False)
        print(A_new)

        # check the local particles
        self.assertEqual(check_array(x_new[:4], [1, 2, 6, 8]),
                         True)
        self.assertEqual(check_array(y_new[:4], [10, 9, 5, 3]),
                         True)
        self.assertEqual(check_array(A_new[:8:2], [1, 2, 6, 8]),
                         True)
        self.assertEqual(check_array(A_new[1:8:2], [1, 2, 6, 8]),
                         True)

        # check the remaining particles
        self.assertEqual(
            check_array(numpy.sort(x_new[4:]),
                        [3, 4, 5, 7, 9, 10]), True
        )
        self.assertEqual(
            check_array(numpy.sort(y_new[4:]),
                        [1, 2, 4, 6, 7, 8]), True
        )
        self.assertEqual(
            check_array(numpy.sort(A_new[8::2]),
                        [3, 4, 5, 7, 9, 10]), True
        )
        self.assertEqual(
            check_array(numpy.sort(A_new[9::2]),
                        [3, 4, 5, 7, 9, 10]), True
        )

        p.set(**{'tag': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
        self.push(p)
        p.align_particles()
        self.pull(p)
        x_new = p.get('x', only_real_particles=False)
        y_new = p.get('y', only_real_particles=False)

        # check the remaining particles
        self.assertEqual(check_array(x_new[:4], [1, 2, 6, 8]),
                         True)
        self.assertEqual(check_array(y_new[:4], [10, 9, 5, 3]),
                         True)
        self.assertEqual(check_array(A_new[:8:2], [1, 2, 6, 8]),
                         True)
        self.assertEqual(check_array(A_new[1:8:2], [1, 2, 6, 8]),
                         True)

        # check the remaining particles
        self.assertEqual(
            check_array(numpy.sort(x_new[4:]),
                        [3, 4, 5, 7, 9, 10]), True
        )
        self.assertEqual(
            check_array(numpy.sort(y_new[4:]),
                        [1, 2, 4, 6, 7, 8]), True
        )
        self.assertEqual(
            check_array(numpy.sort(A_new[8::2]),
                        [3, 4, 5, 7, 9, 10]), True
        )
        self.assertEqual(
            check_array(numpy.sort(A_new[9::2]),
                        [3, 4, 5, 7, 9, 10]), True
        )

    def test_append_parray(self):
        # Given
        p1 = particle_array.ParticleArray(backend=self.backend)
        p1.add_property(**{'name': 'x', 'data': [1, 2, 3]})
        p1.add_property('A', data=2.0, stride=2)
        p2 = particle_array.ParticleArray(x={'data': [4, 5, 6]},
                                          y={'data': [1, 2, 3]},
                                          tag={'data': [1, 0, 1]},
                                          backend=self.backend)

        # When
        p1.append_parray(p2)
        self.pull(p1)

        # Then
        self.assertEqual(p1.get_number_of_particles(), 6)
        self.assertEqual(check_array(p1.x, [1, 2, 3, 5]), True)
        self.assertEqual(check_array(p1.y, [0, 0, 0, 2]), True)
        numpy.testing.assert_array_equal(p1.A, [2.0]*6 + [0.0]*2)
        self.assertEqual(check_array(p1.tag, [0, 0, 0, 0]), True)

        # Given
        p1 = particle_array.ParticleArray(backend=self.backend)
        p1.add_property(**{'name': 'x', 'data': [1, 2, 3]})
        # In this case the new strided prop is in the second parray.
        p2 = particle_array.ParticleArray(x={'data': [4, 5, 6]},
                                          y={'data': [1, 2, 3]},
                                          tag={'data': [1, 0, 1]},
                                          backend=self.backend)
        p2.add_property('A', data=2.0, stride=2)

        # When
        p1.append_parray(p2)
        self.pull(p1)

        # Then
        self.assertEqual(p1.get_number_of_particles(), 6)
        self.assertEqual(check_array(p1.x, [1, 2, 3, 5]), True)
        self.assertEqual(check_array(p1.y, [0, 0, 0, 2]), True)
        self.assertEqual(check_array(p1.A, [0.0]*6 + [2.0]*2), True)
        self.assertEqual(check_array(p1.tag, [0, 0, 0, 0]), True)

    def test_copy_properties(self):
        # Given
        p1 = particle_array.ParticleArray(backend=self.backend)
        p1.add_property(**{'name': 'x',
                           'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        p1.add_property(name='y')
        p1.add_property(name='t')
        p1.add_property('A', data=2.0, stride=2)

        p2 = particle_array.ParticleArray(backend=self.backend)
        p2.add_property(name='t', data=[-1, -1, -1, -1])
        p2.add_property(name='s', data=[2, 3, 4, 5])
        p2.add_property('A', data=3.0, stride=2)

        # When
        p1.copy_properties(p2, start_index=5, end_index=9)

        # Then
        self.assertEqual(check_array(p1.t, [0, 0, 0, 0, 0, -1, -1, -1, -1, 0]),
                         True)
        numpy.testing.assert_array_equal(
            p1.A, [2.0]*10 + [3.0]*8 + [2.0]*2
        )

        # When
        p1.add_property('s')
        p1.copy_properties(p2, start_index=5, end_index=9)

        # Then
        self.assertEqual(check_array(p1.t, [0, 0, 0, 0, 0, -1, -1, -1, -1, 0]),
                         True)
        self.assertEqual(
            check_array(p1.s, [0, 0, 0, 0, 0, 2, 3, 4, 5, 0]), True
        )
        numpy.testing.assert_array_equal(
            p1.A, [2.0]*10 + [3.0]*8 + [2.0]*2
        )

    def test_that_constants_can_be_added(self):
        # Given
        p = particle_array.ParticleArray(backend=self.backend)
        nprop = len(p.properties)
        self.assertEqual(len(p.constants), 0)

        # When
        p.add_constant('s', 0.0)
        p.add_constant('ii', 0)
        p.add_constant('v', [0.0, 1.0, 2.0])

        # Then
        self.assertEqual(len(p.constants), 3)
        self.assertEqual(len(p.properties), nprop)
        self.assertEqual(p.s[0], 0.0)
        self.assertEqual(p.ii[0], 0)
        self.assertTrue(str(p.ii[0].dtype).startswith('int'))
        self.assertTrue(check_array(p.v, [0.0, 1.0, 2.0]))

    def test_that_constants_can_be_set_in_constructor(self):
        # Given
        # When
        p = particle_array.ParticleArray(
            constants=dict(s=0.0, v=[0.0, 1.0, 2.0]),
            backend=self.backend
        )
        nprop = len(p.properties)

        # Then
        self.assertEqual(len(p.constants), 2)
        self.assertEqual(len(p.properties), nprop)
        self.assertEqual(p.s[0], 0.0)
        self.assertTrue(check_array(p.v, [0.0, 1.0, 2.0]))

    def test_that_get_works_on_constants(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)

        # When
        p.add_constant('s', 0.0)
        p.add_constant('v', [0.0, 1.0, 2.0])

        # Then
        self.assertTrue(check_array(p.get('s'), [0.0]))
        self.assertTrue(check_array(p.get('v'), [0.0, 1.0, 2.0]))

    def test_that_constants_are_not_resized_when_particles_are_added(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1.0],
                                         backend=self.backend)

        # When
        p.add_constant('v', [0.0, 1.0])
        p.add_particles(x=[2.0, 3.0])
        self.pull(p)

        # Then
        self.assertTrue(check_array(p.v, [0.0, 1.0]))
        self.assertTrue(check_array(p.x, [1.0, 2.0, 3.0]))

    def test_that_set_works_on_constants(self):
        # Given
        constants = dict(v=[0.0, 0.0, 0.0], c=[0.0, 0.0, 0.0])
        p = particle_array.ParticleArray(name='f', constants=constants,
                                         backend=self.backend)

        # When
        p.set(v=[0.0, 1.0, 2.0])
        p.c = [0.0, 1.0, 2.0]

        # Then
        self.assertEqual(len(p.constants), 2)
        self.assertTrue(check_array(p.get('v'), [0.0, 1.0, 2.0]))
        self.assertTrue(check_array(p.get('c'), [0.0, 1.0, 2.0]))

    def test_that_get_carray_works_with_constants(self):
        # Given
        p = particle_array.ParticleArray(backend=self.backend)
        v = [0.0, 1.0, 2.0]

        # When
        p.add_constant('v', v)
        a = p.get_carray('v')

        # Then
        self.assertEqual(a.get_c_type(), 'double')
        self.assertTrue(check_array(a.get_npy_array(), v))

    def test_empty_clone_works_without_specific_props(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)
        p.set_output_arrays(['x', 'A'])
        v = [0.0, 1.0, 2.0]
        p.add_constant('v', v)

        # When
        clone = p.empty_clone()
        self.pull(clone)

        # Then
        self.assertTrue(check_array(clone.v, v))
        self.assertEqual(sorted(clone.output_property_arrays),
                         sorted(p.output_property_arrays))

    def test_empty_clone_works_with_specific_props(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)
        p.set_output_arrays(['x', 'A'])
        v = [0.0, 1.0, 2.0]
        p.add_constant('v', v)

        # When
        clone = p.empty_clone(props=['x', 'A'])
        self.pull(clone)

        # Then
        self.assertTrue(check_array(clone.v, v))
        self.assertEqual(sorted(clone.output_property_arrays),
                         sorted(p.output_property_arrays))
        self.assertFalse('y' in clone.properties)
        self.assertTrue('x' in clone.properties)
        self.assertTrue('A' in clone.properties)

    def test_extract_particles_works_without_specific_props_without_dest(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)

        # When.
        n = p.extract_particles([1])
        self.pull(n)

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(p.A), 6)
        self.assertEqual(len(n.x), 1)
        self.assertEqual(len(n.A), 2)
        self.assertEqual(n.x[0], 2.0)
        numpy.testing.assert_array_equal(n.A, [2, 3])

    def test_extract_particles_works_with_specific_props_without_dest(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3], y=[0, 0, 0],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)

        # When.
        n = p.extract_particles([1, 2], props=['x', 'A'])
        self.pull(n)

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(n.x), 2)
        self.assertEqual(n.x[0], 2.0)
        self.assertEqual(n.x[0], 2.0)
        numpy.testing.assert_array_equal(n.A, [2, 3, 4, 5])
        self.assertFalse('y' in n.properties)

    def test_extract_particles_works_without_specific_props_with_dest(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)
        n = p.empty_clone()

        # When.
        p.extract_particles([1], dest_array=n)
        self.pull(n)

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(p.A), 6)
        self.assertEqual(len(n.x), 1)
        self.assertEqual(len(n.A), 2)
        self.assertEqual(n.x[0], 2.0)
        numpy.testing.assert_array_equal(n.A, [2, 3])

    def test_extract_particles_works_with_non_empty_dest(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)
        n = p.empty_clone()
        n.extend(2)

        # When.
        p.extract_particles([1], dest_array=n)
        self.pull(n)

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(p.A), 6)
        self.assertEqual(len(n.x), 3)
        self.assertEqual(len(n.A), 6)
        numpy.testing.assert_array_equal(n.x, [0.0, 0.0, 2.0])
        numpy.testing.assert_array_equal(n.A, [0, 0, 0, 0, 2, 3])

    def test_extract_particles_works_with_specific_props_with_dest(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3], y=[0, 0, 0],
                                         backend=self.backend)
        p.add_property('A', data=numpy.arange(6), stride=2)
        n = p.empty_clone(props=['x', 'A'])

        # When.
        p.extract_particles([1, 2], dest_array=n, props=['x', 'A'])
        self.pull(n)

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(n.x), 2)
        self.assertEqual(n.x[0], 2.0)
        self.assertEqual(n.x[0], 2.0)
        numpy.testing.assert_array_equal(n.A, [2, 3, 4, 5])
        self.assertFalse('y' in n.properties)

    def test_that_remove_property_also_removes_output_arrays(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3], y=[0, 0, 0],
                                         backend=self.backend)
        p.add_property('test')
        p.set_output_arrays(['x', 'y', 'test'])

        # When
        p.remove_property('test')

        # Then
        self.assertEqual(p.output_property_arrays, ['x', 'y'])


class ParticleArrayUtils(unittest.TestCase):
    def setUp(self):
        get_config().use_opencl = False

    def test_that_get_particles_info_works(self):
        # Given.
        p = particle_array.ParticleArray(name='f', x=[1, 2, 3])
        p.add_property('A', data=numpy.arange(6), stride=2)
        c = [1.0, 2.0]
        p.add_constant('c', c)

        # When.
        info = utils.get_particles_info([p])
        pas = utils.create_dummy_particles(info)
        dummy = pas[0]

        # Then.
        self.assertTrue(check_array(dummy.c, c))
        self.assertEqual(dummy.name, 'f')
        self.assertTrue('x' in dummy.properties)
        self.assertTrue('A' in dummy.properties)
        self.assertTrue('A' in dummy.stride)
        self.assertEqual(dummy.stride['A'], 2)

    def test_get_particle_array_takes_scalars(self):
        # Given/when
        x = [1, 2, 3, 4]
        data = numpy.diag((2, 2))
        pa = utils.get_particle_array(x=x, y=0, rho=1, data=data)

        # Then
        self.assertTrue(numpy.allclose(x, pa.x))
        self.assertTrue(numpy.allclose(numpy.zeros(4), pa.y))
        self.assertTrue(numpy.allclose(numpy.ones(4), pa.rho))
        self.assertTrue(numpy.allclose(numpy.ravel(data), pa.data))


class ParticleArrayTestCPU(unittest.TestCase, ParticleArrayTest):
    """
    Tests for the particle array class.
    """

    def setUp(self):
        get_config().use_opencl = False
        self.backend = None

    def pull(self, p):
        pass

    def push(self, p):
        pass

    def test_pickle(self):
        """
        Tests the pickle and unpickle functions
        """
        p1 = particle_array.ParticleArray()
        p1.add_property('x', data=numpy.arange(10))
        p1.add_property('y', data=numpy.arange(10))
        p1.add_constant('c', [0.0, 1.0])
        p1.align_particles()

        s = pickle.dumps(p1)
        p2 = pickle.loads(s)

        self.assertEqual(len(p1.x), len(p2.x))
        check_array(p1.x, p2.x)
        self.assertEqual(len(p1.y), len(p2.y))
        check_array(p1.y, p2.y)
        self.assertEqual(len(p1.c), len(p2.c))
        check_array(p1.c, p2.c)

    def test_set(self):
        """
        Tests the set function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data': x}, y={'data': y},
                                         z={'data': z}, m={'data': m},
                                         h={'data': h},
                                         backend=self.backend)

        # set the x array with new values
        p.set(**{'x': [4., 3, 2, 1], 'h': [0.2, 0.2, 0.2, 0.2]})
        self.assertEqual(check_array(p.get('x'), [4., 3, 2, 1]), True)
        self.assertEqual(check_array(p.get('h'), [0.2, 0.2, 0.2, 0.2]), True)

        # trying to set the tags
        p.set(**{'tag': [0, 1, 1, 1]})
        p.align_particles()
        self.pull(p)
        self.assertEqual(
            check_array(p.get('tag', only_real_particles=False), [0, 1, 1, 1]),
            True
        )
        self.assertEqual(check_array(p.get('tag'), [0]), True)

        # try setting array with smaller length array.
        p.set(**{'x': [5, 6, 7]})
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [5, 6, 7, 1]), True)
        # try setting array with longer array.
        self.assertRaises(ValueError, p.set, **{'x': [1., 2, 3, 5, 6]})


class ParticleArrayTestOpenCL(unittest.TestCase, ParticleArrayTest):
    def setUp(self):
        ocl = pytest.importorskip("pyopencl")
        cfg = get_config()
        self.orig_use_double = cfg.use_double
        cfg.use_double = True
        self.backend = 'opencl'

    def tearDown(self):
        get_config().use_double = self.orig_use_double

    def pull(self, p):
        p.gpu.pull()

    def push(self, p):
        p.gpu.push()


class ParticleArrayTestCUDA(unittest.TestCase, ParticleArrayTest):
    def setUp(self):
        cu = pytest.importorskip("pycuda")
        cfg = get_config()
        self.orig_use_double = cfg.use_double
        cfg.use_double = True
        self.backend = 'cuda'

    def tearDown(self):
        get_config().use_double = self.orig_use_double

    def pull(self, p):
        p.gpu.pull()

    def push(self, p):
        p.gpu.push()


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    unittest.main()
