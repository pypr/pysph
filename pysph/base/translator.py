'''Simple Python to C converter.

While this is a fresh implementation, it is highly inspired from

https://github.com/mdipierro/ocl

This code does not use meta and uses the standard ast visitor, it is also
tested and modified to be suitable for use with PySPH.

'''

import ast
import inspect
import re
import sys
from textwrap import dedent, wrap

import numpy as np
from mako.template import Template

from pysph.base.config import get_config
from pysph.base.cython_generator import (
    CodeGenerationError, KnownType, Undefined, all_numeric
)

PY_VER = sys.version_info.major


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


class CStructHelper(object):
    def __init__(self, obj):
        self._use_double = get_config().use_double
        self.parse(obj)

    def _get_public_vars(self):
        data = self.obj.__dict__
        vars = {}
        for name in data:
            if name.startswith('_'):
                continue
            value = data[name]
            if isinstance(value, (int, bool)):
                vars[name] = 'int'
            elif isinstance(value, float):
                vars[name] = 'double'

        return vars

    def parse(self, obj):
        self.name = obj.__class__.__name__
        self.obj = obj
        self.vars = self._get_public_vars()

    def get_array(self):
        f_dtype = np.float64 if self._use_double else np.float32
        types = {'int': np.int32, 'double': f_dtype, 'long': np.int64}
        if len(self.vars) > 0:
            obj = self.obj
            fields = []
            for var in sorted(self.vars):
                fields.append((var, types[self.vars[var]]))
            dtype = np.dtype(fields)
            ary = np.empty(1, dtype)
            for var in self.vars:
                ary[var][0] = getattr(obj, var)
            return ary
        else:
            return None

    def get_code(self):
        template = dedent("""
        typedef struct ${class_name} {
        %for name, type in sorted(vars.items()):
            ${type} ${name};
        %endfor
        } ${class_name};
        """)
        t = Template(text=template)
        return t.render(class_name=self.name, vars=self.vars)


class CConverter(ast.NodeVisitor):
    def __init__(self, detect_type=detect_type, known_types=None):
        self._declares = {}
        self._known = set((
            'M_E', 'M_LOG2E', 'M_LOG10E', 'M_LN2', 'M_LN10',
            'M_PI', 'M_PI_2', 'M_PI_4', 'M_1_PI', 'M_2_PI',
            'M_2_SQRTPI', 'M_SQRT2', 'M_SQRT1_2',
            'INFINITY', 'NAN', 'HUGE_VALF', 'pi'
        ))
        self._name_ctx = (ast.Load, ast.Store)
        self._indent = ''
        self._detect_type = detect_type
        self._known_types = known_types if known_types is not None else {}
        self._class_name = ''
        self._src = ''
        self._ignore_methods = []
        self._replacements = {
            'True': '1', 'False': '0', 'None': 'NULL',
            True: '1', False: '0', None: 'NULL',
        }

    def _body_has_return(self, body):
        return re.search(r'\breturn\b', body) is not None

    def _get_self_type(self):
        return KnownType('%s*' % self._class_name)

    def _get_function_args(self, node):
        node_args = node.args.args
        if PY_VER == 2:
            args = [x.id for x in node_args]
        else:
            args = [x.arg for x in node_args]
        defaults = [ast.literal_eval(x) for x in node.args.defaults]

        # Fill up the call_args dict with the defaults.
        call_args = {}
        for i in range(1, len(defaults) + 1):
            call_args[args[-i]] = defaults[-i]

        # Set the rest to Undefined.
        for i in range(len(args) - len(defaults)):
            call_args[args[i]] = Undefined

        if len(self._class_name) > 0:
            call_args['self'] = self._get_self_type()

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

    def _remove_docstring(self, body):
        if body and isinstance(body[0], ast.Expr) and \
           isinstance(body[0].value, ast.Str):
            return body[1:]
        else:
            return body

    def convert(self, src, ignore_methods=None):
        if ignore_methods is not None:
            self._ignore_methods = ignore_methods
        self._src = src.splitlines()
        code = ast.parse(src)
        result = self.visit(code)
        self._ignore_methods = []
        return result

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

    def get_struct_from_instance(self, obj):
        helper = CStructHelper(obj)
        return helper.get_code() + '\n'

    def parse_instance(self, obj, ignore_methods=None):
        code = self.get_struct_from_instance(obj)
        src = dedent(inspect.getsource(obj.__class__))
        code += self.convert(src, ignore_methods)
        return code

    def parse_function(self, obj):
        src = dedent(inspect.getsource(obj))
        return self.convert(src)

    def visit_Add(self, node):
        return '+'

    def visit_And(self, node):
        return '&&'

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            self.error("Assignments can have only one target.", node)
        left, right = node.targets[0], node.value
        if isinstance(right, ast.Call) and right.func.id == 'declare':
            if not isinstance(right.args[0], ast.Str):
                self.error("Argument to declare should be a string.", node)
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
        code = [self.visit(x) for x in self._remove_docstring(node.body)]
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

        if self._class_name and (node.name.startswith('_') or
                                 node.name in self._ignore_methods):
            return ''

        orig_declares = self._declares
        self._declares = {}
        orig_known = set(self._known)
        if PY_VER == 2:
            self._known.update(x.id for x in node.args.args)
        else:
            self._known.update(x.arg for x in node.args.args)

        args = self._get_function_args(node)
        body = '\n'.join(self._indent_block(self.visit(item))
                         for item in self._remove_docstring(node.body))
        if len(self._class_name) > 0:
            func_name = self._class_name + '_' + node.name
        else:
            func_name = node.name
        if self._body_has_return(body):
            sig = 'double %s(%s)' % (func_name, args)
        else:
            sig = 'void %s(%s)' % (func_name, args)

        declares = self._indent_block(self.get_declarations())
        if len(declares) > 0:
            declares += '\n'

        sig = '\n'.join(wrap(
            sig, width=78, subsequent_indent=' '*4, break_long_words=False
        ))
        self._known = orig_known
        self._declares = orig_declares
        return sig + '\n{\n' + declares + body + '\n}\n'

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
        id = node.id
        if id in self._replacements:
            return self._replacements[id]
        if id not in self._declares and id not in self._known:
            self._declares[id] = 'double %s;' % id
        return id

    def visit_NameConstant(self, node):
        value = node.value
        if value in self._replacements:
            return self._replacements[value]
        else:
            return value

    def visit_Not(self, node):
        return '!'

    def visit_NotEq(self, node):
        return '!='

    def visit_Num(self, node):
        return str(node.n)

    def visit_Or(self, node):
        return '||'

    def visit_Pass(self, node):
        return ';'

    def visit_Return(self, node):
        return 'return %s;' % (self.visit(node.value))

    def visit_Sub(self, node):
        return '-'

    def visit_Str(self, node):
        return r'"%s"' % node.s

    def visit_Subscript(self, node):
        return '%s[%s]' % (
            self.visit(node.value), self.visit(node.slice.value)
        )

    def visit_TryExcept(self, node):
        self.error('Try/except not implemented.', node)

    visit_Try = visit_TryExcept

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


def ocl_detect_type(name, value):
    if isinstance(value, KnownType):
        return value.type
    elif name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return '__global double*'
    else:
        return detect_type(name, value)


class OpenCLConverter(CConverter):
    def __init__(self, detect_type=ocl_detect_type, known_types=None):
        super(OpenCLConverter, self).__init__(detect_type, known_types)

    def _get_self_type(self):
        return KnownType('__global %s*' % self._class_name)
