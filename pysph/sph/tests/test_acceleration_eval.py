
# Standard library imports.
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



class TestEquation(Equation):
    def initialize(self, d_idx, d_rho, d_V):
        d_rho[d_idx] = d_V[d_idx]

    def loop(self, d_idx, d_rho, s_idx, s_m, s_u, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ

    def post_loop(self, d_idx, d_rho, s_idx, s_m, s_V, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


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


class TestAccelerationEval1D(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        n = 10
        dx = 1.0/(n-1)
        x = np.linspace(0, 1, n)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx
        pa = get_particle_array(name='fluid', x=x, h=h, m=m)
        self.pa = pa

    def _make_accel_eval(self, equations):
        arrays = [self.pa]
        kernel = CubicSpline(dim=self.dim)
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel
        )
        comp = SPHCompiler(a_eval, integrator=None)
        comp.compile()
        nnps = NNPS(dim=kernel.dim, particles=arrays)
        nnps.update()
        a_eval.set_nnps(nnps)
        return a_eval

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
