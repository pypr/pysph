# Standard library imports.
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest

# Library imports.
import pytest
import numpy as np

# Local imports.
from pysph.base.config import get_config
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.sph.acceleration_eval import (AccelerationEval,
                                         check_equation_array_properties)
from pysph.sph.basic_equations import SummationDensity
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.sph_compiler import SPHCompiler

from pysph.base.reduce_array import serial_reduce_array


class DummyEquation(Equation):
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

    def test_should_raise_runtime_error_when_invalid_dest_source(self):
        # Given
        f = get_particle_array(name='f')

        # When
        eq = SummationDensity(dest='fluid', sources=['f'])

        # Then
        self.assertRaises(
            RuntimeError,
            check_equation_array_properties,
            eq, [f]
        )

        # When
        eq = SummationDensity(dest='f', sources=['fluid'])

        # Then
        self.assertRaises(
            RuntimeError,
            check_equation_array_properties,
            eq, [f]
        )

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
        eq = DummyEquation(dest='f', sources=['f'])

        # Then
        self.assertRaises(RuntimeError,
                          check_equation_array_properties, eq, [f])

    def test_should_fail_when_src_props_dont_exist(self):
        # Given
        f = get_particle_array(name='f')
        f.add_property('V')
        s = get_particle_array(name='s')

        # When
        eq = DummyEquation(dest='f', sources=['f', 's'])

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
        eq = DummyEquation(dest='f', sources=['f', 's'])

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
        d_au[d_idx] += s_m[s_idx] + s_pid[s_idx] + s_tag[s_idx]

    def post_loop(self, d_idx, d_u, d_au, d_pid):
        d_u[d_idx] = d_au[d_idx] + d_pid[d_idx]


class SimpleReduction(Equation):
    def initialize(self, d_idx, d_au):
        d_au[d_idx] = 0.0

    def reduce(self, dst):
        dst.total_mass[0] = serial_reduce_array(dst.m, op='sum')
        if dst.gpu is not None:
            dst.gpu.push('total_mass')


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
        equations = [FindTotalMass(dest='fluid', sources=['fluid'])]
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
            equations=[
                SimpleEquation(dest='fluid', sources=['fluid']),
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
                    equations=[SimpleEquation(dest='fluid', sources=['fluid'])]
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


class EqWithTime(Equation):
    def initialize(self, d_idx, d_au, t, dt):
        d_au[d_idx] = t + dt

    def loop(self, d_idx, d_au, s_idx, s_m, t, dt):
        d_au[d_idx] += t + dt


class TestAccelerationEval1DGPU(unittest.TestCase):
    # Fix this to be a subclass of TestAccelerationEval1D

    def setUp(self):
        self.dim = 1
        n = 10
        dx = 1.0/(n-1)
        x = np.linspace(0, 1, n)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.05
        pa = get_particle_array(name='fluid', x=x, h=h, m=m)
        self.pa = pa

    def _make_accel_eval(self, equations, cache_nnps=True):
        pytest.importorskip('pysph.base.gpu_nnps')
        from pysph.base.gpu_nnps import ZOrderGPUNNPS as GPUNNPS
        arrays = [self.pa]
        kernel = CubicSpline(dim=self.dim)
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel,
            backend='opencl'
        )
        comp = SPHCompiler(a_eval, integrator=None)
        comp.compile()
        self.sph_compiler = comp
        nnps = GPUNNPS(dim=kernel.dim, particles=arrays, cache=cache_nnps)
        nnps.update()
        a_eval.set_nnps(nnps)
        return a_eval

    def test_accel_eval_should_work_on_gpu(self):
        # Given
        pa = self.pa
        equations = [SimpleEquation(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])
        pa.gpu.pull('u')
        self.assertListEqual(list(pa.u), list(expect))

    def test_precomputed_should_work_on_gpu(self):
        # Given
        pa = self.pa
        equations = [SummationDensity(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([7.357, 9.0, 9., 9., 9., 9., 9., 9.,  9.,  7.357])
        pa.gpu.pull('rho')

        print(pa.rho, pa.gpu.rho)
        self.assertTrue(np.allclose(expect, pa.rho, atol=1e-2))

    def test_precomputed_should_work_on_gpu_with_double(self):
        orig = get_config().use_double

        def _cleanup():
            get_config().use_double = orig
        get_config().use_double = True
        self.addCleanup(_cleanup)
        # Given
        pa = self.pa
        equations = [SummationDensity(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([7.357, 9.0, 9., 9., 9., 9., 9., 9.,  9.,  7.357])
        pa.gpu.pull('rho')

        print(pa.rho, pa.gpu.rho)
        self.assertTrue(np.allclose(expect, pa.rho, atol=1e-2))

    def test_equation_with_time_should_work_on_gpu(self):
        # Given
        pa = self.pa
        equations = [EqWithTime(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.2, 0.1)

        # Then
        expect = np.asarray([4., 5., 6., 6., 6., 6., 6., 6.,  5.,  4.])*0.3
        pa.gpu.pull('au')
        print(pa.au, expect)
        self.assertTrue(np.allclose(expect, pa.au))

    def test_update_nnps_is_called_for_opencl(self):
        # Given
        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid']),
                ],
                update_nnps=True
            ),
            Group(
                equations=[EqWithTime(dest='fluid', sources=['fluid'])]
            ),
        ]

        # When
        a_eval = self._make_accel_eval(equations)

        # Then
        h = a_eval.c_acceleration_eval.helper
        assert len(h.calls) == 5
        call = h.calls[0]
        assert call['type'] == 'kernel'
        assert call['method'].function_name == 'g0_fluid_initialize'
        assert call['loop'] is False

        call = h.calls[1]
        assert call['type'] == 'kernel'
        assert call['method'].function_name == 'g0_fluid_on_fluid_loop'
        assert call['loop'] is True

        call = h.calls[2]
        assert call['type'] == 'method'
        assert call['method'] == 'update_nnps'

        call = h.calls[3]
        assert call['type'] == 'kernel'
        assert call['method'].function_name == 'g1_fluid_initialize'
        assert call['loop'] is False

        call = h.calls[4]
        assert call['type'] == 'kernel'
        assert call['method'].function_name == 'g1_fluid_on_fluid_loop'
        assert call['loop'] is True

    def test_should_stop_iteration_with_max_iteration_on_gpu(self):
        pa = self.pa

        class SillyEquation(Equation):
            def loop(self, d_idx, d_au, s_idx, s_m):
                d_au[d_idx] += s_m[s_idx]

            def converged(self):
                return 0

        equations = [Group(
            equations=[
                Group(
                    equations=[SillyEquation(dest='fluid', sources=['fluid'])]
                ),
                Group(
                    equations=[SillyEquation(dest='fluid', sources=['fluid'])]
                ),
            ],
            iterate=True, max_iterations=2,
        )]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])*4.0
        pa.gpu.pull('au')
        self.assertListEqual(list(pa.au), list(expect))

    def test_should_stop_iteration_with_converged_on_gpu(self):
        pa = self.pa

        class SillyEquation1(Equation):
            def __init__(self, dest, sources):
                super(SillyEquation1, self).__init__(dest, sources)
                self.conv = 0

            def loop(self, d_idx, d_au, s_idx, s_m):
                d_au[d_idx] += s_m[s_idx]

            def post_loop(self, d_idx, d_au):
                if d_au[d_idx] > 19.0:
                    self.conv = 1

            def converged(self):
                return self.conv

        equations = [Group(
            equations=[
                Group(
                    equations=[SillyEquation1(dest='fluid', sources=['fluid'])]
                ),
                Group(
                    equations=[SillyEquation1(dest='fluid', sources=['fluid'])]
                ),
            ],
            iterate=True, max_iterations=10,
        )]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.asarray([3., 4., 5., 5., 5., 5., 5., 5.,  4.,  3.])*6.0
        pa.gpu.pull('au')
        self.assertListEqual(list(pa.au), list(expect))

    def test_should_handle_helper_functions_on_gpu(self):
        pa = self.pa

        def helper(x=1.0):
            return x*1.5

        class SillyEquation2(Equation):
            def initialize(self, d_idx, d_au, d_m):
                d_au[d_idx] += helper(d_m[d_idx])

            def _get_helpers_(self):
                return [helper]

        equations = [SillyEquation2(dest='fluid', sources=['fluid'])]

        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.ones(10)*1.5
        pa.gpu.pull('au')
        self.assertListEqual(list(pa.au), list(expect))

    def test_should_run_reduce_when_using_gpu(self):
        # Given.
        pa = self.pa
        pa.add_constant('total_mass', 0.0)
        equations = [SimpleReduction(dest='fluid', sources=['fluid'])]
        a_eval = self._make_accel_eval(equations)

        # When
        a_eval.compute(0.1, 0.1)

        # Then
        expect = np.sum(pa.m)
        pa.gpu.pull('total_mass')
        self.assertAlmostEqual(pa.total_mass[0], expect, 14)

    def test_get_equations_with_converged(self):
        pytest.importorskip('pysph.base.gpu_nnps')
        from pysph.sph.acceleration_eval_opencl_helper import \
            get_equations_with_converged
        # Given
        se = SimpleEquation(dest='fluid', sources=['fluid'])
        se1 = SimpleEquation(dest='fluid', sources=['fluid'])
        sd = SummationDensity(dest='fluid', sources=['fluid'])
        me = MixedTypeEquation(dest='fluid', sources=['fluid'])
        eq_t = EqWithTime(dest='fluid', sources=['fluid'])
        g = Group(
            equations=[
                Group(equations=[Group(equations=[se, sd])],
                      iterate=True, max_iterations=10),
                Group(equations=[me, eq_t, se1]),
            ],
        )

        # When
        eqs = get_equations_with_converged(g)

        # Then
        assert eqs == [se, se1]
