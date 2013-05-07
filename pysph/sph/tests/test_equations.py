
# Standard library imports.
import numpy
from textwrap import dedent
import unittest

# Local imports.
from pysph.sph.equations import (BasicCodeBlock, CodeBlock, Context, Equation, 
    Group, VariableClashError, sort_precomputed)


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
        keys = c.keys()
        keys.sort()
        self.assertEqual(keys, ['a', 'b'])
        values = c.values()
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


class TestCodeBlock(TestBase):
    
    def check_code_block(self, eq):
        self.assert_seq_equal(eq.symbols, ['d_x', 'd_idx', 'XIJ', 'WIJ'])
        self.assert_seq_equal(eq.src_arrays, [])
        self.assert_seq_equal(eq.dest_arrays, ['d_x'])
        ctx = eq.context
        self.assertTrue('s_idx' not in ctx)
        self.assertEqual(ctx.d_idx, 0)
        self.assertEqual(ctx.XIJ, [0.0, 0.0, 0.0])
        self.assertEqual(ctx.HIJ, 0.0)
        x = numpy.zeros(2, dtype=float)
        self.assertTrue(numpy.alltrue(ctx.d_x == x))
        pre = eq.precomputed.keys()
        self.assert_seq_equal(pre, ['XIJ', 'HIJ', 'WIJ'])
    
        # It should be easy to evaluate the equation.    
        res = eq()
        self.assertEqual(sum(res.d_x), 0)
        res = eq(XIJ=[1,2,1], WIJ=10)
        self.assertEqual(sum(res.d_x), 21)
        
    def test_basic_usage(self):
        code = '''
        d_x[d_idx] = XIJ[0] + WIJ*XIJ[1]
        '''
        eq = CodeBlock(code=code)        
        self.check_code_block(eq)        


class TestEquations(TestBase):
    def test_simple_equation(self):
        eq = Equation('fluid')
        self.assertEqual(eq.name, 'Equation')
        self.assertEqual(eq.no_source, True)
        self.assertEqual(eq.dest, 'fluid')
        self.assertEqual(eq.sources, None)
        self.assertEqual(eq.loop, None)
        self.assertEqual(eq.post_loop, None)
        
        eq = Equation('fluid', sources=['fluid'], name='Test')
        self.assertEqual(eq.name, 'Test')
        self.assertEqual(eq.no_source, False)
        self.assertEqual(eq.dest, 'fluid')
        self.assertEqual(eq.sources, ['fluid'])
        
        class Test(Equation):
            pass
        eq = Test('fluid')
        self.assertEqual(eq.name, 'Test')
        self.assertEqual(eq.no_source, True)
        
    def test_continuity_equation(self):
        from pysph.sph.equations import ContinuityEquation
        e = ContinuityEquation(dest='fluid', sources=['fluid'])
        # Call the loop code.
        r = e.loop()
        self.assertEqual(r.vijdotdwij, 0)
        self.assertEqual(r.d_arho[0], 0.0)
        self.assertEqual(r.d_arho[1], 0.0)
        # Now call with specific arguments.
        r = e.loop(DWIJ=[1,1,1], VIJ=[1,1,1], s_m=[1,1])
        self.assertEqual(r.vijdotdwij, 3)
        self.assertEqual(r.d_arho[0], 3.0)
        self.assertEqual(r.d_arho[1], 0.0)
        
    def test_order_of_precomputed(self):
        pre_comp = CodeBlock.pre_comp
        input = dict((x, pre_comp[x]) for x in ['WIJ', 'DWIJ', 'XIJ', 'HIJ'])
        pre = sort_precomputed(input)
        self.assertEqual(pre.keys(), ['HIJ', 'XIJ', 'DWIJ', 'WIJ'])


class TestEq1(Equation):
    def setup(self):
        code = dedent("""
        x = WIJ
        """)

        self.loop = CodeBlock(code=code, x=0.0)
        self.post_loop = CodeBlock(code='x = d_h[d_idx]', x=0.0)
        
class TestEq2(Equation):
    def setup(self):
        code = dedent("""
        x = s_idx + d_idx
        """)

        self.loop = CodeBlock(code=code, x=1)

class TestEq3(Equation):
    def setup(self):
        code = dedent("""
        x += s_idx
        """)
        self.initialize_vars = set(('x'))
        self.loop = CodeBlock(code=code, x=1)

class TestGroup(TestBase):
    def setUp(self):
        from pysph.sph.equations import SummationDensity, TaitEOS, Group
        self.group = Group([SummationDensity('f', ['f']), TaitEOS('f', None)])

    def test_precomputed(self):
        g = self.group
        self.assertEqual(len(g.precomputed), 2)
        self.assertEqual(g.precomputed.keys(), ['HIJ', 'WIJ'])
        
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
        expect = ['B', 'gamma', 'gamma1', 'rho0', 'c0', 'tmp', 'ratio', 'WIJ', 
                  'HIJ']
        self.assert_seq_equal(names, expect)
        # Should work without error.
        g.check_variables(names)

    def test_variable_clash(self):
        e1 = TestEq1('f', ['f'])
        e2 = TestEq2('f', ['f'])
        g = Group([e1, e2])
        names = g.get_variable_names()
        self.assertRaises(VariableClashError, g.check_variables, names)
        
        # Now set the type correctly but a different value and exception should
        # be raised.
        e2.loop.context.x = 1.0
        self.assertRaises(VariableClashError, g.check_variables, names)
        
    def test_array_declarations(self):
        g = self.group
        expect = 'cdef double* d_x'
        self.assertEqual(g.get_array_declarations(['d_x']), expect)
        
    def test_variable_declarations(self):
        g = self.group
        context = Context(x=1.0)
        expect = 'cdef double x = 1.0'
        self.assertEqual(g.get_variable_declarations(context), expect)
        context = Context(x=1)
        expect = 'cdef long x = 1'
        self.assertEqual(g.get_variable_declarations(context), expect)

        context = Context(x=[1., 2.])
        expect = 'cdef double[2] x'
        self.assertEqual(g.get_variable_declarations(context), expect)
    
        context = Context(x=(0, 1., 2.))
        expect = 'cdef double[3] x'
        self.assertEqual(g.get_variable_declarations(context), expect)

    def test_variable_initializations(self):
        g = self.group
        context = Context(x=1.0)
        expect = ''
        self.assertEqual(g.get_variable_initializations(context), expect)
        # No initialization for arrays currently.
        context = Context(x=(0, 1., 2.))
        expect = ''
        self.assertEqual(g.get_variable_initializations(context), expect)

        context = Context(x=1)
        g = Group([TestEq3('f', ['f'])])
        expect = 'x = 1'
        self.assertEqual(g.get_variable_initializations(context), expect)
        
    def test_loop_code(self):
        from pysph.base.kernels import CubicSpline
        k = CubicSpline(dim=3)
        e1 = TestEq1('f', ['f'])
        e2 = TestEq2('f', ['f'])
        g = Group([e1, e2])
        result = g.get_loop_code(k)
        expect = dedent('''\
            HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])
            WIJ = CubicSplineKernel(d_x[d_idx], d_y[d_idx], d_z[d_idx], s_x[s_idx], s_y[s_idx], s_z[s_idx], HIJ)

            # TestEq1.
            x = WIJ
            # TestEq2.
            x = s_idx + d_idx
            ''')
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(result, expect, msg)

    def test_post_loop_code(self):
        from pysph.base.kernels import CubicSpline
        k = CubicSpline(dim=3)
        e1 = TestEq1('f', ['f'])
        e2 = TestEq2('f', ['f'])
        g = Group([e1, e2])
        result = g.get_post_loop_code(k)
        expect = dedent('''\
            # TestEq1.
            x = d_h[d_idx]
            ''')
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(result, expect, msg)

        e2.post_loop = CodeBlock(code='x = x + 1', x=0.0)
        g = Group([e1, e2])
        result = g.get_post_loop_code(k)
        expect = dedent('''\
            # TestEq1.
            x = d_h[d_idx]
            # TestEq2.
            x = x + 1
            ''')
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(result, expect, msg)


if __name__ == '__main__':
    unittest.main()