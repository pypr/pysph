
import ast
from textwrap import dedent
import unittest

from pysph.base.ast_utils import (get_assigned, get_aug_assign_symbols, get_symbols,
    has_node, has_return)


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

    def test_get_aug_assign_symbols(self):
        code = '''
        x = 1
        '''
        result = list(get_aug_assign_symbols(dedent(code)))
        result.sort()
        expect = []
        self.assertEqual(result, expect)

        code = '''
        x += 1
        d_x[d_idx] += s_x[s_idx]
        '''
        result = list(get_aug_assign_symbols(dedent(code)))
        result.sort()
        expect = ['d_x', 'x']
        self.assertEqual(result, expect)

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


if __name__ == '__main__':
    unittest.main()
