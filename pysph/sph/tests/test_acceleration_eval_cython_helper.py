# Standard library imports.
import unittest

# Library imports.
import numpy as np

from compyle.api import get_config, set_config

# Local library imports.
from pysph.base.particle_array import ParticleArray
from compyle.api import KnownType
from pysph.base.kernels import CubicSpline
from pysph.sph.acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays,
    AccelerationEvalCythonHelper
)
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.equation import Group


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


class TestAccelerationHelperCython(unittest.TestCase):
    def setUp(self):
        cfg = get_config()
        cfg.use_openmp = True

    def tearDown(self):
        set_config(None)

    def test_parallel_range_unsets_chunksize_on_start_stop_idx(self):
        # Given
        pa = ParticleArray(name='f', m=[1.0], rho=[0.0])

        eqs = [Group(
            equations=[SummationDensity(dest='f', sources=['f'])],
            start_idx=1, stop_idx=2
        )]
        aeval = AccelerationEval([pa], eqs, kernel=CubicSpline(dim=1))

        # When
        helper = AccelerationEvalCythonHelper(aeval)
        result = helper.get_parallel_range(eqs[0])

        # Then
        expect = ("prange(D_START_IDX, NP_DEST, 1, schedule='dynamic', "
                  "nogil=True)")
        self.assertEqual(result, expect)

    def test_parallel_range_without_loop_count(self):
        # Given
        pa = ParticleArray(name='f', m=[1.0], rho=[0.0])

        eqs = [Group(
            equations=[SummationDensity(dest='f', sources=['f'])],
        )]
        aeval = AccelerationEval([pa], eqs, kernel=CubicSpline(dim=1))

        # When
        helper = AccelerationEvalCythonHelper(aeval)
        result = helper.get_parallel_range(eqs[0])

        # Then
        expect = ("prange(D_START_IDX, NP_DEST, 1, schedule='dynamic', "
                  "chunksize=64, nogil=True)")
        self.assertEqual(result, expect)

        result = helper.get_parallel_range(eqs[0], nogil=False)

        # Then
        expect = ("prange(D_START_IDX, NP_DEST, 1, schedule='dynamic', "
                  "chunksize=64)")
        self.assertEqual(result, expect)
