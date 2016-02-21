# Standard library imports.
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest

# Library imports.
import numpy as np

# Local imports.
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.sph.acceleration_eval import (AccelerationEval,
    check_equation_array_properties)
from pysph.sph.basic_equations import SummationDensity
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.sph_compiler import SPHCompiler

from pysph.base.reduce_array import serial_reduce_array


class TestEquation(Equation):
    def initialize(self, d_idx, d_rho, d_V):
        d_rho[d_idx] = d_V[d_idx]

    def loop(self, d_idx, d_rho, s_idx, s_m, s_u, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ

    def post_loop(self, d_idx, d_rho, s_idx, s_m, s_V, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


class FindTotalMass(Equation):
    def initialize(self, d_idx, d_m, d_total_mass):
        # FIXME: This is stupid and should be fixed if we add a separate
        # initialize_once function or so.
        d_total_mass[0] = 0.0

    def post_loop(self, d_idx, d_m, d_total_mass):
        d_total_mass[0] += d_m[d_idx]


class TestCheckEquationArrayProps(unittest.TestCase):

    def test_should_pass_when_properties_exist(self):
        # Given
        f = get_particle_array(name='f')

        # When
        eq = SummationDensity(dest='f', sources=['f'])

        # Then
        check_equation_array_properties(eq, [f])

    def test_should_fail_when_props_dont_exist(self):
        # Given
        f = get_particle_array(name='f')

        # When
        eq = TestEquation(dest='f', sources=['f'])

        # Then
        self.assertRaises(RuntimeError,
                          check_equation_array_properties, eq, [f])

    def test_should_fail_when_src_props_dont_exist(self):
        # Given
        f = get_particle_array(name='f')
        f.add_property('V')
        s = get_particle_array(name='s')

        # When
        eq = TestEquation(dest='f', sources=['f', 's'])

        # Then
        self.assertRaises(RuntimeError,
                          check_equation_array_properties, eq, [f, s])

    def test_should_pass_when_src_props_exist(self):
        # Given
        f = get_particle_array(name='f')
        f.add_property('V')
        s = get_particle_array(name='s')
        s.add_property('V')

        # When
        eq = TestEquation(dest='f', sources=['f', 's'])

        # Then
        check_equation_array_properties(eq, [f, s])

    def test_should_check_constants(self):
        # Given
        f = get_particle_array(name='f')

        # When
        eq = FindTotalMass(dest='f', sources=['f'])

        # Then.
        self.assertRaises(RuntimeError,
                          check_equation_array_properties, eq, [f])

        # When.
        f.add_constant('total_mass', 0.0)

        # Then.
        check_equation_array_properties(eq, [f])


class SimpleEquation(Equation):
    def __init__(self, dest, sources):
        super(SimpleEquation, self).__init__(dest, sources)
        self.count = 0

    def initialize(self, d_idx, d_u, d_au):
        d_u[d_idx] = 0.0
        d_au[d_idx] = 0.0

    def loop(self, d_idx, d_au, s_idx, s_m):
        #print d_idx, s_idx
        d_au[d_idx] += s_m[s_idx]

    def post_loop(self, d_idx, d_u, d_au):
        d_u[d_idx] = d_au[d_idx]

    def converged(self):
        self.count += 1
        result = self.count - 1
        if result > 0:
            # Reset the count for the next loop.
            self.count = 0
        return result

class MixedTypeEquation(Equation):
    def initialize(self, d_idx, d_u, d_au, d_pid, d_tag):
        d_u[d_idx] = 0.0 + d_pid[d_idx]
        d_au[d_idx] = 0.0 + d_tag[d_idx]

    def loop(self, d_idx, d_au, s_idx, s_m, s_pid, s_tag):
        #print d_idx, s_idx
        d_au[d_idx] += s_m[s_idx] + s_pid[s_idx] + s_tag[s_idx]

    def post_loop(self, d_idx, d_u, d_au, d_pid):
        d_u[d_idx] = d_au[d_idx] + d_pid[d_idx]


class SimpleReduction(Equation):
    def reduce(self, dst):
        dst.total_mass[0] = serial_reduce_array(dst.array.m, op='sum')


class TestAccelerationEval1D(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        n = 10
        dx = 1.0/(n-1)
        x = np.linspace(0, 1, n)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.05
        pa = get_particle_array(name='fluid', x=x, h=h, m=m)
        self.pa = pa

    def _make_accel_eval(self, equations, cache_nnps=False):
        arrays = [self.pa]
        kernel = CubicSpline(dim=self.dim)
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel
        )
        comp = SPHCompiler(a_eval, integrator=None)
        comp.compile()
        nnps = NNPS(dim=kernel.dim, particles=arrays, cache=cache_nnps)
        nnps.update()
        a_eval.set_nnps(nnps)
        return a_eval

    def test_should_support_constants(self):
        # Given
        pa = self.pa
        pa.add_constant('total_mass', 0.0)
        equations = [FindTotalMass (dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        self.assertEqual(pa.total_mass, 10.0)

    def test_should_not_iterate_normal_group(self):
        # Given
        pa = self.pa
        equations = [SimpleEquation(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])
        self.assertListEqual(list(pa.u), list(expect))

    def test_should_work_with_cached_nnps(self):
        # Given
        pa = self.pa
        equations = [SimpleEquation(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations, cache_nnps=True)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])
        self.assertListEqual(list(pa.u), list(expect))

    def test_should_iterate_iterated_group(self):
        # Given
        pa = self.pa
        equations = [Group(
            equations=[SimpleEquation(dest='fluid', sources=['fluid']),
                       SimpleEquation(dest='fluid', sources=['fluid']),
                      ],
            iterate=True
        )]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])*2
        self.assertListEqual(list(pa.u), list(expect))

    def test_should_iterate_nested_groups(self):
        pa = self.pa
        equations = [Group(
            equations=[
                Group(
                    equations=[SimpleEquation(dest='fluid', sources=['fluid'])]
                ),
                Group(
                    equations=[SimpleEquation(dest='fluid', sources=['fluid'])],
                ),
            ],
            iterate=True,
        )]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])
        self.assertListEqual(list(pa.u), list(expect))

    def test_should_run_reduce(self):
        # Given.
        pa = self.pa
        pa.add_constant('total_mass', 0.0)
        equations = [SimpleReduction(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.sum(pa.m)
        self.assertAlmostEqual(pa.total_mass[0], expect, 14)

    def test_should_work_with_non_double_arrays(self):
        # Given
        pa = self.pa
        equations = [MixedTypeEquation(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])
        self.assertListEqual(list(pa.u), list(expect))
