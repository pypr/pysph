
# Standard library imports.
import numpy
from textwrap import dedent
import unittest

# Local imports.
from pysph.base.cython_generator import KnownType
from pysph.sph.equation import (BasicCodeBlock, Context, Equation,
    Group, sort_precomputed)


class TestContext(unittest.TestCase):
    def test_basic_usage(self):
        c = Context(a=1, b=2)
        self.assertEqual(c.a, 1)
        self.assertEqual(c.b, 2)
        self.assertEqual(c['a'], 1)
        self.assertEqual(c['b'], 2)
        c.c = 3
        self.assertEqual(c.c, 3)
        self.assertEqual(c['c'], 3)

    def test_context_behaves_like_dict(self):
        c = Context(a=1)
        c.b = 2
        keys = list(c.keys())
        keys.sort()
        self.assertEqual(keys, ['a', 'b'])
        values = list(c.values())
        values.sort()
        self.assertEqual(values, [1, 2])
        self.assertTrue('a' in c)
        self.assertTrue('b' in c)
        self.assertTrue('c' not in c)


class TestBase(unittest.TestCase):
    def assert_seq_equal(self, got, expect):
        g = list(got)
        g.sort()
        e = list(expect)
        e.sort()
        self.assertEqual(g, e, 'got %s, expected %s'%(g, e))

class TestBasicCodeBlock(TestBase):

    def test_basic_code_block(self):
        code = '''
        x = 1
        d_x[d_idx] += s_x[s_idx] + x
        '''
        cb = BasicCodeBlock(code=code)
        expect = ['d_idx', 'd_x', 's_idx', 's_x', 'x']
        self.assert_seq_equal(cb.symbols, expect)
        self.assert_seq_equal(cb.src_arrays, ['s_x'])
        self.assert_seq_equal(cb.dest_arrays, ['d_x'])
        ctx = cb.context
        self.assertEqual(ctx.s_idx, 0)
        self.assertEqual(ctx.d_idx, 0)
        x = numpy.zeros(2, dtype=float)
        self.assertTrue(numpy.alltrue(ctx.d_x == x))
        self.assertTrue(numpy.alltrue(ctx.s_x == x))

    def test_that_code_block_is_callable(self):
        code = '''
        x = 1
        d_x[d_idx] += s_x[s_idx] + x
        '''
        cb = BasicCodeBlock(code=code)
        # The code block should be callable.
        res = cb()
        self.assertTrue(sum(res.d_x) == 1)

        # Should take arguments to update the context.
        x = numpy.ones(2, dtype=float)*10
        res = cb(s_x=x)
        self.assertTrue(sum(res.d_x) == 11)


class TestEquations(TestBase):
    def test_simple_equation(self):
        eq = Equation('fluid', None)
        self.assertEqual(eq.name, 'Equation')
        self.assertEqual(eq.no_source, True)
        self.assertEqual(eq.dest, 'fluid')
        self.assertEqual(eq.sources, None)
        self.assertFalse(hasattr(eq, 'loop'))
        self.assertFalse(hasattr(eq, 'post_loop'))
        self.assertFalse(hasattr(eq, 'initialize'))

        eq = Equation('fluid', sources=['fluid'])
        self.assertEqual(eq.name, 'Equation')
        self.assertEqual(eq.no_source, False)
        self.assertEqual(eq.dest, 'fluid')
        self.assertEqual(eq.sources, ['fluid'])

        class Test(Equation):
            pass
        eq = Test('fluid', [])
        self.assertEqual(eq.name, 'Test')
        self.assertEqual(eq.no_source, True)

    def test_continuity_equation(self):
        from pysph.sph.basic_equations import ContinuityEquation
        e = ContinuityEquation(dest='fluid', sources=['fluid'])
        # Call the loop code.

        d_arho = [0.0, 0.0, 0.0]
        s_m = [0.0, 0.0]
        r = e.loop(d_idx=0, d_arho=d_arho, s_idx=0, s_m=s_m,
                   DWIJ=[0,0,0], VIJ=[0,0,0])
        self.assertEqual(d_arho[0], 0.0)
        self.assertEqual(d_arho[1], 0.0)
        # Now call with specific arguments.
        s_m = [1, 1]
        r = e.loop(d_idx=0, d_arho=d_arho, s_idx=0, s_m=s_m,
                   DWIJ=[1,1,1], VIJ=[1,1,1])
        self.assertEqual(d_arho[0], 3.0)
        self.assertEqual(d_arho[1], 0.0)

    def test_order_of_precomputed(self):
        try:
            pre_comp = Group.pre_comp
            pre_comp.AIJ = BasicCodeBlock(code=dedent("""
                AIJ[0] = XIJ[0]/RIJ
                AIJ[1] = XIJ[1]/RIJ
                AIJ[2] = XIJ[2]/RIJ
                """),
                AIJ=[1.0, 0.0, 0.0])
            input = dict((x, pre_comp[x]) for x in ['RIJ', 'R2IJ',
                                                    'XIJ', 'HIJ', 'AIJ'])
            pre = sort_precomputed(input)
            self.assertEqual(
                list(pre.keys()), ['HIJ', 'XIJ', 'R2IJ', 'RIJ', 'AIJ']
            )
        finally:
            from pysph.sph.equation import precomputed_symbols
            Group.pre_comp = precomputed_symbols()


class TestEq1(Equation):
    def loop(self, WIJ=0.0):
        x = WIJ

    def post_loop(self, d_idx, d_h):
        x = d_h[d_idx]

class TestEq2(Equation):
    def loop(self, d_idx, s_idx):
        x = s_idx + d_idx

class TestGroup(TestBase):
    def setUp(self):
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.wc.basic import TaitEOS
        self.group = Group(
            [SummationDensity('f', ['f']),
             TaitEOS('f', None, rho0=1.0, c0=1.0, gamma=1.4, p0=1.0)]
        )

    def test_precomputed(self):
        g = self.group
        self.assertEqual(len(g.precomputed), 5)
        self.assertEqual(list(g.precomputed.keys()),
                         ['HIJ', 'XIJ', 'R2IJ', 'RIJ', 'WIJ'])

    def test_array_names(self):
        g = self.group
        src, dest = g.get_array_names()
        s_ex = ['s_m', 's_x', 's_y', 's_z', 's_h']
        d_ex = ['d_rho', 'd_p', 'd_h', 'd_cs', 'd_x', 'd_y', 'd_z']
        self.assert_seq_equal(src, s_ex)
        self.assert_seq_equal(dest, d_ex)

    def test_variable_names(self):
        g = self.group
        names = g.get_variable_names()
        expect = ['WIJ', 'RIJ', 'R2IJ', 'XIJ', 'HIJ']
        self.assert_seq_equal(names, expect)

    def test_array_declarations(self):
        g = self.group
        expect = 'cdef double* d_x'
        self.assertEqual(g.get_array_declarations(['d_x']), expect)

    def test_array_declarations_with_known_types(self):
        # Given
        g = self.group
        known_types = {'d_x': KnownType('float*')}
        # When
        result = g.get_array_declarations(['d_x'], known_types)
        # Then.
        expect = 'cdef float* d_x'
        self.assertEqual(result, expect)

    def test_variable_declarations(self):
        g = self.group
        context = Context(x=1.0)
        expect = 'cdef double x = 1.0'
        self.assertEqual(g.get_variable_declarations(context), expect)
        context = Context(x=1)
        expect = 'cdef long x = 1'
        self.assertEqual(g.get_variable_declarations(context), expect)

        context = Context(x=[1., 2.])
        expect = ('cdef DoubleArray _x = DoubleArray(aligned(2, 8)*self.n_threads)\n'
                  'cdef double* x = _x.data')
        self.assertEqual(g.get_variable_declarations(context), expect)

        context = Context(x=(0, 1., 2.))
        expect = ('cdef DoubleArray _x = DoubleArray(aligned(3, 8)*self.n_threads)\n'
                  'cdef double* x = _x.data')
        self.assertEqual(g.get_variable_declarations(context), expect)

    def test_loop_code(self):
        from pysph.base.kernels import CubicSpline
        k = CubicSpline(dim=3)
        e1 = TestEq1('f', ['f'])
        e2 = TestEq2('f', ['f'])
        g = Group([e1, e2])
        # First get the equation wrappers so the equation names are setup.
        w = g.get_equation_wrappers()
        result = g.get_loop_code(k)
        expect = dedent('''\
            HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])
            XIJ[0] = d_x[d_idx] - s_x[s_idx]
            XIJ[1] = d_y[d_idx] - s_y[s_idx]
            XIJ[2] = d_z[d_idx] - s_z[s_idx]
            R2IJ = XIJ[0]*XIJ[0] + XIJ[1]*XIJ[1] + XIJ[2]*XIJ[2]
            RIJ = sqrt(R2IJ)
            WIJ = self.kernel.kernel(XIJ, RIJ, HIJ)

            self.test_eq10.loop(WIJ)
            self.test_eq20.loop(d_idx, s_idx)
            ''')
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(result, expect, msg)

    def test_post_loop_code(self):
        from pysph.base.kernels import CubicSpline
        k = CubicSpline(dim=3)
        e1 = TestEq1('f', ['f'])
        e2 = TestEq2('f', ['f'])
        g = Group([e1, e2])
        # First get the equation wrappers so the equation names are setup.
        w = g.get_equation_wrappers()
        result = g.get_post_loop_code(k)
        expect = dedent('''\
            self.test_eq10.post_loop(d_idx, d_h)
            ''')
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(result, expect, msg)

if __name__ == '__main__':
    unittest.main()
