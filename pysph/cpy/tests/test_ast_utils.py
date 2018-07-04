
import ast
import sys
from textwrap import dedent
import unittest

from ..ast_utils import (
    get_assigned, get_symbols, get_unknown_names_and_calls,
    has_node, has_return
)


class TestASTUtils(unittest.TestCase):
    def test_get_symbols(self):
        code = '''
        x = 1
        d_x[d_idx] += s_x[s_idx]
        '''
        tree = ast.parse(dedent(code))
        result = list(get_symbols(tree))
        result.sort()
        expect = ['d_idx', 'd_x', 's_idx', 's_x', 'x']
        self.assertEqual(result, expect)

        # Test if it parses with the code itself instead of a tree.
        result = list(get_symbols(dedent(code)))
        result.sort()
        self.assertEqual(result, expect)

        result = list(get_symbols(tree, ctx=ast.Store))
        result.sort()
        self.assertEqual(result, ['x'])

    def test_has_return(self):
        code = dedent('''
                x = 1
                ''')
        self.assertFalse(has_return(code))
        code = dedent('''
                def f():
                    pass
                ''')
        self.assertFalse(has_return(code))
        code = dedent('''
                def f(x):
                    return x+1
                ''')
        self.assertTrue(has_return(code))

    def test_has_node(self):
        code = dedent('''
                x = 1
                ''')
        self.assertFalse(has_node(code, (ast.Return, ast.AugAssign)))
        code = dedent('''
                def f():
                    pass
                ''')
        self.assertTrue(has_node(code, (ast.AugAssign, ast.FunctionDef)))

    def test_assigned_values(self):
        code = dedent('''
            u[0] = 0.0
            x = 1
            y = sin(x)*theta
            z += 1
            ''')
        assigned = list(sorted(get_assigned(code)))
        # sin or theta should not be detected.
        expect = ['u', 'x', 'y', 'z']
        self.assertEqual(assigned, expect)

    def test_assigned_tuple_expansion(self):
        code = dedent('''
            u, v = 0.0, 1.0
            [x, y] = 0.0, 1.0
            ''')
        assigned = list(sorted(get_assigned(code)))
        expect = ['u', 'v', 'x', 'y']
        self.assertEqual(assigned, expect)

    def test_get_unknown_names_and_calls(self):
        code = dedent('''
        def f(x):
            g(h(x))
            y = x + SIZE
            for i in range(y):
                x += func(JUNK)
            sin(x)
        ''')

        # When
        names, calls = get_unknown_names_and_calls(code)

        # Then.
        e_names = {'SIZE', 'i', 'JUNK'}
        e_calls = {'g', 'h', 'range', 'func', 'sin'}
        self.assertSetEqual(names, e_names)
        self.assertSetEqual(calls, e_calls)

    @unittest.skipIf(sys.version_info < (3, 4),
                     reason='Test requires Python 3.')
    def test_get_unknown_names_and_calls_with_py3_annotation(self):
        code = dedent('''
        from pysph.cpy import types as T

        def f(x: T.doublep, n: T.int_)-> T.double:
            s = declare('double')
            for i in range(n):
                s += func(x)
            return s
        ''')

        # When
        names, calls = get_unknown_names_and_calls(code)

        # Then.
        e_names = {'i'}
        e_calls = {'declare', 'func', 'range'}
        self.assertSetEqual(names, e_names)
        self.assertSetEqual(calls, e_calls)


if __name__ == '__main__':
    unittest.main()
