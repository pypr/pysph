'''Simple Python to C converter.

While this is a fresh implementation, it is highly inspired from

https://github.com/mdipierro/ocl

This code does not use meta and uses the standard ast visitor, it is also
tested and modified to be suitable for use with PySPH.


TODO
----

- support declaring variables using the declare function.
- support user-guided type declarations.
- support some automatic type detection using default args.
- support simple classes -> mangled methods + structs.

'''

import ast
import re


def py2c(src):
    code = ast.parse(src)
    print(ast.dump(code))
    converter = CConverter()
    result = converter.visit(code)
    r = converter.get_declarations() + result
    print(r)
    return r


class CConverter(ast.NodeVisitor):
    def __init__(self):
        self._declares = {}
        self._known = set()
        self._name_ctx = (ast.Load, ast.Store)
        self._indent = ''

    def _body_has_return(self, body):
        return re.search(r'\breturn\b', body) is not None

    def _indent_block(self, code):
        lines = code.splitlines()
        pad = ' '*4
        return '\n'.join(pad + x for x in lines)

    def get_declarations(self):
        if len(self._declares) > 0:
            return '\n'.join(
                sorted(self._declares.values())
            ) + '\n'
        else:
            return ''

    def visit_Name(self, node):
        assert isinstance(node.ctx, self._name_ctx)
        if node.id not in self._declares and node.id not in self._known:
            self._declares[node.id] = 'double %s;' % node.id
        return node.id

    def visit_Add(self, node):
        return '+'

    def visit_And(self, node):
        return '&&'

    def visit_Assign(self, node):
        return '%s = %s;' % (self.visit(node.targets[0]), self.visit(node.value))

    def visit_Attribute(self, node):
        return '%s->%s' % (self.visit(node.value), node.attr)

    def visit_AugAssign(self, node):
        return '%s %s= %s;' % (self.visit(node.target), self.visit(node.op),
                               self.visit(node.value))

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return 'pow(%s, %s)' %(self.visit(node.left), self.visit(node.right))
        else:
            result = tuple(self.visit(x) for x in (node.left, node.op, node.right))
            return '(%s %s %s)' % result

    def visit_BoolOp(self, node):
        op = ' %s ' % self.visit(node.op)
        return '(%s)' % (op.join(self.visit(x) for x in node.values))

    def visit_Break(self, node):
        return 'break;'

    def visit_Call(self, node):
        return '{func}({args})'.format(
            func=node.func.id,
            args=', '.join(self.visit(x) for x in node.args)
        )

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError('Only simple comparisons are allowed.')
        return '(%s %s %s)' % (self.visit(node.left),
                               self.visit(node.ops[0]),
                               self.visit(node.comparators[0]))

    def visit_Continue(self, node):
        return 'continue;'

    def visit_Div(self, node):
        return '/'

    def visit_Eq(self, node):
        return '=='

    def visit_Expr(self, node):
        return self.visit(node.value) + ';'

    def visit_For(self, node):
        if node.iter.func.id != 'range':
            raise NotImplementedError('Only for var in range syntax supported.')
        if node.orelse:
            raise NotImplementedError('For/else not supported.')
        args = node.iter.args
        if len(args) == 1:
            start, stop, incr = 0, self.visit(args[0]), 1
        elif len(args) == 2:
            start, stop, incr = self.visit(args[0]), self.visit(args[1]), 1
        elif len(args) == 3:
            start, stop, incr = [self.visit(x) for x in args]
        else:
            raise NotImplementedError('range should have either [1,2,3] args')
        if isinstance(node.target, ast.Name) and \
           not node.target.id in self._known:
            self._known.add(node.target.id)
        r = ('for (long {i}={start}; {i}<{stop}; {i}+={incr})'
             ' {{\n{block}\n}}\n').format(
                 i=self.visit(node.target), start=start, stop=stop, incr=incr,
                 block='\n'.join(
                     self._indent_block(self.visit(x)) for x in node.body
                 )
             )
        return r

    def visit_FunctionDef(self, node):
        orig_known = set(self._known)
        self._known.update(x.id for x in node.args.args)
        args = ', '.join('double %s' % x.id for x in node.args.args)
        body = '\n'.join('    %s' % self.visit(item) for item in node.body)
        if self._body_has_return(body):
            sig = 'double %s(%s) {\n' % (node.name, args)
        else:
            sig = 'void %s(%s) {\n' % (node.name, args)

        self._known = orig_known
        return sig + body + '\n}'

    def visit_Gt(self, node):
        return '>'

    def visit_GtE(self, node):
        return '>='

    def visit_If(self, node):
        code = 'if ({cond}) {{\n{block}\n}}\n'.format(
            cond=self.visit(node.test),
            block='\n'.join(
                self._indent_block(self.visit(x)) for x in node.body
            )
        )
        if node.orelse:
            code += 'else {{\n{block}\n}}\n'.format(
                block='\n'.join(
                    self._indent_block(self.visit(x)) for x in node.orelse
                )
            )
        return code

    def visit_Is(self, node):
        return '=='

    def visit_IsNot(self, node):
        return '!='

    def visit_Lt(self, node):
        return '<'

    def visit_LtE(self, node):
        return '<='

    def visit_Mod(self, node):
        return '%'

    def visit_Module(self, node):
        return '\n'.join(
            self.visit(item) for item in node.body
        )

    def visit_Mult(self, node):
        return '*'

    def visit_Not(self, node):
        return '!'

    def visit_NotEq(self, node):
        return '!='

    def visit_Num(self, node):
        return node.n

    def visit_Or(self, node):
        return '||'

    def visit_Pass(self, node):
        return ';'

    def visit_Return(self, node):
        return 'return %s;' % (self.visit(node.value))

    def visit_Sub(self, node):
        return '-'

    def visit_Subscript(self, node):
        return '%s[%s]' % (self.visit(node.value), self.visit(node.slice.value))

    def visit_TryExcept(self, node):
        raise NotImplementedError('Try/except not implemented.')

    def visit_UnaryOp(self, node):
        return '(%s %s)' % (self.visit(node.op), self.visit(node.operand))

    def visit_USub(self, node):
        return '-'

    def visit_While(self, node):
        if node.orelse:
            raise NotImplementedError('Does not support while/else clauses.')
        return 'while ({cond}) {{\n{block}\n}}\n'.format(
            cond=self.visit(node.test),
            block='\n'.join(
                self._indent_block(self.visit(x)) for x in node.body
            )
        )
