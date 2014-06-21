
import unittest

# Local imports.
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.sph.sph_eval import check_equation_array_properties
from pysph.sph.basic_equations import SummationDensity

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
