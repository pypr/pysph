"""
Tests for the particle array module.
"""

# standard imports
import unittest
import numpy

# local imports
import pysph
from pysph.base import particle_array
from pysph.base import utils

from pyzoltan.core import carray
from pyzoltan.core.carray import LongArray, IntArray, DoubleArray

import pickle

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

###############################################################################
# `ParticleArrayTest` class.
###############################################################################
class ParticleArrayTest(unittest.TestCase):
    """
    Tests for the particle array class.
    """
    def test_constructor(self):
        """
        Test the constructor.
        """
        # Default constructor test.
        p = particle_array.ParticleArray(name='test_particle_array')

        self.assertEqual(p.name, 'test_particle_array')
        self.assertEqual(p.is_dirty, True)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual(p.properties['tag'].length, 0)

        # Constructor with some properties.
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        p = particle_array.ParticleArray(x={'data':x},
                                         y={'data':y},
                                         z={'data':z},
                                         m={'data':m},
                                         h={'data':h})

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
        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z},
                                         tag={'data':tags,'type':'int'})
        self.assertEqual(check_array(p.get('tag', only_real_particles=False),
                                     [0,0,1,1]), True)
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [1,3,2,4]), True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [0,2,1,3]), True)
        self.assertEqual(check_array(p.get('z', only_real_particles=False),
                                     [0,0,0,0]), True)

        # trying to create particle array without any values but some
        # properties.
        p = particle_array.ParticleArray(x={}, y={}, z={}, h={})

        self.assertEqual(p.get_number_of_particles(), 0)
        self.assertEqual('x' in p.properties, True)
        self.assertEqual('y' in p.properties, True)
        self.assertEqual('z' in p.properties, True)
        self.assertEqual('tag' in p.properties, True)

        # now trying to supply some properties with values and others without
        p = particle_array.ParticleArray(x={'default':10.0}, y={'data':[1.0, 2.0]},
                                         z={}, h={'data':[0.1, 0.1]})

        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [10., 10.]), True)
        self.assertEqual(check_array(p.y, [1., 2.]), True)
        self.assertEqual(check_array(p.z, [0, 0]), True)
        self.assertEqual(check_array(p.h, [0.1, 0.1]), True)

    def test_constructor_works_with_simple_props(self):
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        p = particle_array.ParticleArray(x=x, y=y, name='fluid')
        self.assertEqual(p.name, 'fluid')

        self.assertEqual('x' in p.properties, True)
        self.assertEqual('y' in p.properties, True)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual('pid' in p.properties, True)
        self.assertEqual('gid' in p.properties, True)

        # get the properties are check if they are the same
        self.assertEqual(check_array(p.x, x), True)
        self.assertEqual(check_array(p.y, y), True)

    def test_get_number_of_particles(self):
        """
        Tests the get_number_of_particles of particles.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m}, h={'data':h})

        self.assertEqual(p.get_number_of_particles(), 4)

    def test_get(self):
        """
        Tests the get function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m}, h={'data':h})

        self.assertEqual(check_array(x, p.get('x')), True)
        self.assertEqual(check_array(y, p.get('y')), True)
        self.assertEqual(check_array(z, p.get('z')), True)
        self.assertEqual(check_array(m, p.get('m')), True)
        self.assertEqual(check_array(h, p.get('h')), True)

    def test_set(self):
        """
        Tests the set function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})

        # set the x array with new values
        p.set(**{'x':[4., 3, 2, 1], 'h':[0.2, 0.2, 0.2, 0.2]})
        self.assertEqual(check_array(p.get('x'), [4., 3, 2, 1]), True)
        self.assertEqual(check_array(p.get('h'), [0.2, 0.2, 0.2, 0.2]), True)

        # trying to set the tags
        p.set(**{'tag':[0, 1, 1, 1]})
        self.assertEqual(check_array(p.get('tag', only_real_particles=False)
                                     , [0, 1, 1, 1]), True)
        self.assertEqual(check_array(p.get('tag'), [0]), True)

        # try setting array with smaller length array.
        p.set(**{'x':[5, 6, 7]})
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [5, 6, 7, 1]), True)
        # try setting array with longer array.
        self.assertRaises(ValueError, p.set, **{'x':[1., 2, 3, 5, 6]})

    def test_clear(self):
        """
        Tests the clear function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})

        p.clear()

        self.assertEqual(len(p.properties), 3)
        self.assertEqual('tag' in p.properties, True)
        self.assertEqual(p.properties['tag'].length, 0)

        self.assertEqual('pid' in p.properties, True)
        self.assertEqual(p.properties['pid'].length, 0)

        self.assertEqual('gid' in p.properties, True)
        self.assertEqual(p.properties['gid'].length, 0)

        self.assertEqual(p.is_dirty, True)

    def test_getattr(self):
        """
        Tests the __getattr__ function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})

        self.assertEqual(check_array(x, p.x), True)
        self.assertEqual(check_array(y, p.y), True)
        self.assertEqual(check_array(z, p.z), True)
        self.assertEqual(check_array(m, p.m), True)
        self.assertEqual(check_array(h, p.h), True)

        # try getting an non-existant attribute
        self.assertRaises(AttributeError, p.__getattr__, 'a')

    def test_setattr(self):
        """
        Tests the __setattr__ function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})

        p.x = p.x*2.0

        self.assertEqual(check_array(p.get('x'), [2., 4, 6, 8]), True)
        p.x = p.x + 3.0*p.x
        self.assertEqual(check_array(p.get('x'), [8., 16., 24., 32.]), True)

    def test_remove_particles(self):
        """
        Tests the remove_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})
        remove_arr = LongArray(0)
        remove_arr.append(0)
        remove_arr.append(1)

        p.remove_particles(remove_arr)

        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)

        # now try invalid operations to make sure errors are raised.
        remove_arr.resize(10)
        self.assertRaises(ValueError, p.remove_particles, remove_arr)

        # now try to remove a particle with index more that particle
        # length.
        remove_arr = [2]

        p.remove_particles(remove_arr)
        # make sure no change occurred.
        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)

    def test_add_particles(self):
        """
        Tests the add_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})
        p.set_dirty(False)

        new_particles = {}
        new_particles['x'] = numpy.array([5., 6, 7])
        new_particles['y'] = numpy.array([4., 5, 6])
        new_particles['z'] = numpy.array([0., 0, 0])

        p.add_particles(**new_particles)

        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(p.is_dirty, True)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)

        p.set_dirty(False)

        # try adding an empty particle list
        p.add_particles(**{})
        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(p.is_dirty, False)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)

        # adding particles with tags
        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h})
        p.add_particles(x=[5, 6, 7, 8], tag=[1, 1, 0, 0])

        self.assertEqual(p.get_number_of_particles(), 8)
        self.assertEqual(check_array(p.x, [1, 2, 3, 4, 7, 8]), True)
        self.assertEqual(check_array(p.y, [0, 1, 2, 3, 0, 0]), True)
        self.assertEqual(check_array(p.z, [0, 0, 0, 0, 0, 0]), True)

    def test_remove_tagged_particles(self):
        """
        Tests the remove_tagged_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        tag = [1, 1, 1, 0]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h}, tag={'data':tag})

        p.remove_tagged_particles(0)

        self.assertEqual(p.get_number_of_particles(), 3)
        self.assertEqual(check_array(p.get('x', only_real_particles=False)
                                     , [1, 2, 3.]), True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False)
                                     , [0., 1, 2]), True)
        self.assertEqual(check_array(p.get('z', only_real_particles=False)
                                     , [0., 0, 0]), True)
        self.assertEqual(check_array(p.get('h', only_real_particles=False)
                                     , [.1, .1, .1]), True)
        self.assertEqual(check_array(p.get('m', only_real_particles=False)
                                     , [1., 1., 1.]), True)

        self.assertEqual(check_array(p.x, []), True)
        self.assertEqual(check_array(p.y, []), True)
        self.assertEqual(check_array(p.z, []), True)
        self.assertEqual(check_array(p.h, []), True)
        self.assertEqual(check_array(p.m, []), True)

    def test_add_property(self):
        """
        Tests the add_property function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        tag = [0, 0, 0, 0]

        p = particle_array.ParticleArray(x={'data':x}, y={'data':y},
                                         z={'data':z}, m={'data':m},
                                         h={'data':h}, tag={'data':tag})

        p.add_property(**{'name':'x'})
        # make sure the current 'x' property is intact.
        self.assertEqual(check_array(p.x, x), True)

        # add a property with complete specification
        p.add_property(**{'name':'f1',
                        'data':[1, 1, 2, 3],
                        'type':'int',
                        'default':4})
        self.assertEqual(check_array(p.f1, [1, 1, 2, 3]), True)
        self.assertEqual(type(p.properties['f1']), IntArray)
        self.assertEqual(p.default_values['f1'], 4)

        # add a property without specifying the type
        p.add_property(**{'name':'f2',
                        'data':[1, 1, 2, 3],
                        'default':4.0})
        self.assertEqual(type(p.properties['f2']), DoubleArray)
        self.assertEqual(check_array(p.f2, [1, 1, 2, 3]), True)

        p.add_property(**{'name':'f3'})
        self.assertEqual(type(p.properties['f3']), DoubleArray)
        self.assertEqual(p.properties['f3'].length, p.get_number_of_particles())
        self.assertEqual(check_array(p.f3, [0, 0, 0, 0]), True)

        p.add_property(**{'name':'f4', 'default':3.0})
        self.assertEqual(type(p.properties['f4']), DoubleArray)
        self.assertEqual(p.properties['f4'].length, p.get_number_of_particles())
        self.assertEqual(check_array(p.f4, [3, 3, 3, 3]), True)

    def test_extend(self):
        """
        Tests the extend function.
        """
        p = particle_array.ParticleArray(default_particle_tag=10, x={},
                                         y={'default':-1.})
        p.extend(5)

        self.assertEqual(p.get_number_of_particles(), 5)
        self.assertEqual(check_array(p.get(
                    'x', only_real_particles=False), [0, 0, 0, 0, 0]), True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [-1., -1., -1., -1., -1.]), True)
        self.assertEqual(check_array(p.get('tag', only_real_particles=False),
                                     [10, 10, 10, 10, 10]), True)

    def test_align_particles(self):
        """
        Tests the align particles function.
        """
        p = particle_array.ParticleArray()
        p.add_property(**{'name':'x', 'data':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        p.add_property(**{'name':'y', 'data':[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
        p.set_dirty(False)
        p.set(**{'tag':[0, 0, 1, 1, 1, 0, 4, 0, 1, 5]})
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]),
                         True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)

        self.assertEqual(p.is_dirty, True)

        p.set_dirty(False)
        p.set(**{'tag':[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
        self.assertEqual(check_array(p.get('x', only_real_particles=False),
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]),
                         True)
        self.assertEqual(check_array(p.get('y', only_real_particles=False),
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)
        self.assertEqual(p.is_dirty, False)

    def test_append_parray(self):
        """
        Tests the append_parray function.
        """
        p1 = particle_array.ParticleArray()
        p1.add_property(**{'name':'x', 'data':[1, 2, 3]})
        p1.align_particles()

        p2 = particle_array.ParticleArray(x={'data':[4, 5, 6]},
                                          y={'data':[1, 2, 3 ]},
                                          tag={'data':[1, 0, 1]})


        p1.append_parray(p2)

        # print(p1.get('x', only_real_particles=False))
        # print(p1.get('y', only_real_particles=False))
        # print(p1.get('tag', only_real_particles=False))

        self.assertEqual(p1.get_number_of_particles(), 6)
        self.assertEqual(check_array(p1.x, [1, 2, 3, 5]), True)
        self.assertEqual(check_array(p1.y, [0, 0, 0, 2]), True)
        self.assertEqual(check_array(p1.tag, [0, 0, 0, 0]), True)

    def test_copy_properties(self):
        """
        Tests the copy properties function.
        """
        p1 = particle_array.ParticleArray()
        p1.add_property(**{'name':'x', 'data':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        p1.add_property(**{'name':'y'})
        p1.add_property(**{'name':'t'})
        p1.align_particles()

        p2 = particle_array.ParticleArray()
        p2.add_property(**{'name':'t', 'data':[-1, -1, -1, -1]})
        p2.add_property(**{'name':'s', 'data':[2, 3, 4, 5]})
        p2.align_particles()

        p1.copy_properties(p2, start_index=5, end_index=9)
        self.assertEqual(check_array(p1.t, [0, 0, 0, 0, 0, -1, -1, -1, -1, 0]),
                         True)

        p1.add_property(**{'name':'s'})
        p1.copy_properties(p2, start_index=5, end_index=9)
        self.assertEqual(check_array(p1.t, [0, 0, 0, 0, 0, -1, -1, -1, -1, 0]),
                         True)
        self.assertEqual(check_array(p1.s, [0, 0, 0, 0, 0, 2, 3, 4, 5, 0]), True)

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

    def test_that_constants_can_be_added(self):
        # Given
        p = particle_array.ParticleArray()
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
            constants=dict(s=0.0, v=[0.0, 1.0, 2.0])
        )
        nprop = len(p.properties)

        # Then
        self.assertEqual(len(p.constants), 2)
        self.assertEqual(len(p.properties), nprop)
        self.assertEqual(p.s[0], 0.0)
        self.assertTrue(check_array(p.v, [0.0, 1.0, 2.0]))

    def test_that_get_works_on_constants(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1,2,3])

        # When
        p.add_constant('s', 0.0)
        p.add_constant('v', [0.0, 1.0, 2.0])

        # Then
        self.assertTrue(check_array(p.get('s'), [0.0]))
        self.assertTrue(check_array(p.get('v'), [0.0, 1.0, 2.0]))

    def test_that_constants_are_not_resized_when_particles_are_added(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1.0])

        # When
        p.add_constant('v', [0.0, 1.0])
        p.add_particles(x=[2.0, 3.0])

        # Then
        self.assertTrue(check_array(p.v, [0.0, 1.0]))
        self.assertTrue(check_array(p.x, [1.0, 2.0, 3.0]))

    def test_that_set_works_on_constants(self):
        # Given
        constants = dict(v=[0.0, 0.0, 0.0], c=[0.0, 0.0, 0.0])
        p = particle_array.ParticleArray(name='f', constants=constants)

        # When
        p.set(v=[0.0, 1.0, 2.0])
        p.c = [0.0, 1.0, 2.0]

        # Then
        self.assertEqual(len(p.constants), 2)
        self.assertTrue(check_array(p.get('v'), [0.0, 1.0, 2.0]))
        self.assertTrue(check_array(p.get('c'), [0.0, 1.0, 2.0]))

    def test_that_get_carray_works_with_constants(self):
        # Given
        p = particle_array.ParticleArray()
        v = [0.0, 1.0, 2.0]

        # When
        p.add_constant('v', v)
        a = p.get_carray('v')

        # Then
        self.assertEqual(a.get_c_type(), 'double')
        self.assertTrue(check_array(a.get_npy_array(), v))

    def test_extract_particles_extracts_particles_and_output_arrays(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1,2,3])
        p.set_output_arrays(['x'])

        # When.
        n = p.extract_particles(indices=[1])

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(n.x), 1)
        self.assertEqual(n.x[0], 2.0)
        self.assertEqual(n.output_property_arrays, p.output_property_arrays)

    def test_extract_particles_works_with_specific_props(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1,2,3], y=[0,0,0])
        p.set_output_arrays(['x', 'y'])

        # When.
        n = p.extract_particles(indices=[1], props=['x'])

        # Then.
        self.assertEqual(len(p.x), 3)
        self.assertEqual(len(n.x), 1)
        self.assertEqual(n.x[0], 2.0)
        self.assertFalse('y' in n.properties)
        self.assertEqual(sorted(p.output_property_arrays), sorted(['x', 'y']))
        self.assertEqual(n.output_property_arrays, ['x'])

    def test_that_remove_property_also_removes_output_arrays(self):
        # Given
        p = particle_array.ParticleArray(name='f', x=[1,2,3], y=[0,0,0])
        p.add_property('test')
        p.set_output_arrays(['x', 'y', 'test'])

        # When
        p.remove_property('test')

        # Then
        self.assertEqual(p.output_property_arrays, ['x', 'y'])


class ParticleArrayUtils(unittest.TestCase):
    def test_that_get_particles_info_works(self):
        # Given.
        p = particle_array.ParticleArray(name='f', x=[1,2,3])
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


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    unittest.main()
