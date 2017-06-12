'''Simple Python to C converter.

While this is a fresh implementation, it is highly inspired from

https://github.com/mdipierro/ocl

This code does not use meta and uses the standard ast visitor, it is also
tested and modified to be suitable for use with PySPH.


TODO
----

- convert class instance to struct.

'''

import ast
import re
from pysph.base.cython_generator import (
    CodeGenerationError, KnownType, Undefined, all_numeric
)


def detect_type(name, value):
    if isinstance(value, KnownType):
        return value.type
    if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return 'double*'
    if name in ['s_idx', 'd_idx']:
        return 'long'
    if value is Undefined or isinstance(value, Undefined):
        raise CodeGenerationError('Unknown type, for %s' % name)

    if isinstance(value, bool):
        return 'int'
    elif isinstance(value, int):
        return 'long'
    elif isinstance(value, float):
        return 'double'
    elif isinstance(value, (list, tuple)):
        if all_numeric(value):
            # We don't deal with integer lists for now.
            return 'double*'
        else:
            raise CodeGenerationError(
                'Unknown type, for %s with value %s' % (name, value)
            )
    else:
        raise CodeGenerationError(
            'Unknown type, for %s with value %s' % (name, value)
        )


def py2c(src, detect_type=detect_type, known_types=None):
    converter = CConverter(detect_type=detect_type, known_types=known_types)
    result = converter.convert(src)
    r = converter.get_declarations() + result
    print(r)
    return r


class CConverter(ast.NodeVisitor):
    def __init__(self, detect_type=detect_type, known_types=None):
        self._declares = {}
        self._known = set()
        self._name_ctx = (ast.Load, ast.Store)
        self._indent = ''
        self._detect_type = detect_type
        self._known_types = known_types if known_types is not None else {}
        self._class_name = ''
        self._src = ''

    def _body_has_return(self, body):
        return re.search(r'\breturn\b', body) is not None

    def _get_function_args(self, node):
        node_args = node.args.args
        args = [x.id for x in node_args]
        defaults = [ast.literal_eval(x) for x in node.args.defaults]

        # Fill up the call_args dict with the defaults.
        call_args = {}
        for i in range(1, len(defaults) + 1):
            call_args[args[-i]] = defaults[-i]

        # Set the rest to Undefined.
        for i in range(len(args) - len(defaults)):
            call_args[args[i]] = Undefined

        if len(self._class_name) > 0:
            call_args['self'] = KnownType('%s*' % self._class_name)

        call_args.update(self._known_types)
        call_sig = []
        for arg in args:
            value = call_args[arg]
            type = self._detect_type(arg, value)
            call_sig.append('{type} {arg}'.format(type=type, arg=arg))

        return ', '.join(call_sig)

    def _get_variable_declaration(self, type_str, name):
        if type_str.startswith('matrix'):
            shape = ast.literal_eval(type_str[7:-1])
            sz = ''.join('[%d]' % x for x in shape)
            return 'double %s%s;' % (name, sz)
        else:
            return '%s %s;' % (type_str, name)

    def _indent_block(self, code):
        lines = code.splitlines()
        pad = ' '*4
        return '\n'.join(pad + x for x in lines)

    def convert(self, src):
        self._src = src.splitlines()
        code = ast.parse(src)
        return self.visit(code)

    def error(self, message, node):
        msg = '\nError in code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' '*node.col_offset + '^' + '\n\n'
        msg += message
        raise NotImplementedError(msg)

    def get_declarations(self):
        if len(self._declares) > 0:
            return '\n'.join(
                sorted(self._declares.values())
            ) + '\n'
        else:
            return ''

    def visit_Add(self, node):
        return '+'

    def visit_And(self, node):
        return '&&'

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Assignments can have only one target."
        left, right = node.targets[0], node.value
        if isinstance(right, ast.Call) and right.func.id == 'declare':
            assert isinstance(right.args[0], ast.Str), \
                "Argument to declare should be a string."
            type = right.args[0].s
            self._known.add(left.id)
            return self._get_variable_declaration(type, self.visit(left))
        return '%s = %s;' % (self.visit(left), self.visit(right))

    def visit_Attribute(self, node):
        return '%s->%s' % (self.visit(node.value), node.attr)

    def visit_AugAssign(self, node):
        return '%s %s= %s;' % (self.visit(node.target), self.visit(node.op),
                               self.visit(node.value))

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return 'pow(%s, %s)' % (
                self.visit(node.left), self.visit(node.right)
            )
        else:
            result = tuple(self.visit(x)
                           for x in (node.left, node.op, node.right))
            return '(%s %s %s)' % result

    def visit_BoolOp(self, node):
        op = ' %s ' % self.visit(node.op)
        return '(%s)' % (op.join(self.visit(x) for x in node.values))

    def visit_Break(self, node):
        return 'break;'

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            return '{func}({args})'.format(
                func=node.func.id,
                args=', '.join(self.visit(x) for x in node.args)
            )
        elif (isinstance(node.func, ast.Attribute) and
              len(self._class_name) > 0):
            return '{func}({args})'.format(
                func='%s_%s' % (self._class_name, node.func.attr),
                args='self, ' + ', '.join(self.visit(x) for x in node.args)
            )
        else:
            self.error('Unsupported function call', node)

    def visit_ClassDef(self, node):
        self._class_name = node.name
        # FIXME: Does not handle base class methods.
        code = [self.visit(x) for x in node.body]
        self._class_name = ''
        return '\n'.join(code)

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            self.error('Only simple comparisons are allowed.', node)
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
            self.error(
                'Only for var in range syntax supported.', node.iter
            )
        if node.orelse:
            self.error('For/else not supported.', node.orelse[0])
        args = node.iter.args
        if len(args) == 1:
            start, stop, incr = 0, self.visit(args[0]), 1
        elif len(args) == 2:
            start, stop, incr = self.visit(args[0]), self.visit(args[1]), 1
        elif len(args) == 3:
            start, stop, incr = [self.visit(x) for x in args]
        else:
            self.error('range should have either [1,2,3] args', node.iter)
        if isinstance(node.target, ast.Name) and \
           node.target.id not in self._known:
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
        assert node.args.vararg is None, \
            "Functions with varargs nor supported in line %d." % node.lineno
        assert node.args.kwarg is None, \
            "Functions with kwargs not supported in line %d." % node.lineno

        orig_known = set(self._known)
        self._known.update(x.id for x in node.args.args)
        args = self._get_function_args(node)
        body = '\n'.join('    %s' % self.visit(item) for item in node.body)
        if len(self._class_name) > 0:
            func_name = self._class_name + '_' + node.name
        else:
            func_name = node.name
        if self._body_has_return(body):
            sig = 'double %s(%s) {\n' % (func_name, args)
        else:
            sig = 'void %s(%s) {\n' % (func_name, args)

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

    def visit_Name(self, node):
        assert isinstance(node.ctx, self._name_ctx)
        if node.id not in self._declares and node.id not in self._known:
            self._declares[node.id] = 'double %s;' % node.id
        return node.id

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
        return '%s[%s]' % (
            self.visit(node.value), self.visit(node.slice.value)
        )

    def visit_TryExcept(self, node):
        self.error('Try/except not implemented.', node)

    def visit_UnaryOp(self, node):
        return '(%s %s)' % (self.visit(node.op), self.visit(node.operand))

    def visit_USub(self, node):
        return '-'

    def visit_While(self, node):
        if node.orelse:
            self.error('Does not support while/else clauses.', node.orelse[0])
        return 'while ({cond}) {{\n{block}\n}}\n'.format(
            cond=self.visit(node.test),
            block='\n'.join(
                self._indent_block(self.visit(x)) for x in node.body
            )
        )
