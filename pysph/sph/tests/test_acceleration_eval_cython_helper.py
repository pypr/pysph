# Standard library imports.
import unittest

# Library imports.
import numpy as np

# Local library imports.
from pysph.base.particle_array import ParticleArray
from pysph.base.cython_generator import KnownType
from pysph.sph.acceleration_eval_cython_helper import (get_all_array_names,
    get_known_types_for_arrays)


class TestGetAllArrayNames(unittest.TestCase):
    def test_that_all_properties_are_found(self):
        x = np.linspace(0, 1, 10)
        pa = ParticleArray(name='f', x=x)
        result = get_all_array_names([pa])
        self.assertEqual(len(result), 3)
        self.assertEqual(result['DoubleArray'], set(('x',)))
        self.assertEqual(result['IntArray'], set(('pid', 'tag')))
        self.assertEqual(result['UIntArray'], set(('gid',)))

    def test_that_all_properties_are_found_with_multiple_arrays(self):
        x = np.linspace(0, 1, 10)
        pa1 = ParticleArray(name='f', x=x)
        pa2 = ParticleArray(name='b', y=x)
        result = get_all_array_names([pa1, pa2])
        self.assertEqual(len(result), 3)
        self.assertEqual(result['DoubleArray'], set(('x', 'y')))
        self.assertEqual(result['IntArray'], set(('pid', 'tag')))
        self.assertEqual(result['UIntArray'], set(('gid',)))

class TestGetKnownTypesForAllArrays(unittest.TestCase):
    def test_that_all_types_are_detected_correctly(self):
        x = np.linspace(0, 1, 10)
        pa = ParticleArray(name='f', x=x)
        pa.remove_property('pid')
        info = get_all_array_names([pa])
        result = get_known_types_for_arrays(info)

        expect = {'d_gid': KnownType("unsigned int*"),
         'd_tag': KnownType("int*"),
         'd_x': KnownType("double*"),
         's_gid': KnownType("unsigned int*"),
         's_tag': KnownType("int*"),
         's_x': KnownType("double*")}
        for key in expect:
            self.assertEqual(repr(result[key]), repr(expect[key]))

