from functools import wraps
from textwrap import dedent, wrap

from mako.template import Template
from pytools import memoize_method
import numpy as np
import inspect
import ast
import importlib

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed, \
        filter_calls, BUILTINS
from .types import dtype_to_ctype, annotate, get_declare_info
from .parallel import Elementwise
from .extern import Extern

import pysph.cpy.array as array


def get_ctype_from_arg(arg):
    if isinstance(arg, array.Array):
        return arg.gptr_type
    elif isinstance(arg, np.ndarray):
        return dtype_to_ctype(arg.dtype)
    else:
        return False


class JITHelper(ast.NodeVisitor):
    def __init__(self, func):
        self.func = func
        self.calls = set()
        self.arg_types = {}

    def annotate(self, *args):
        func, self.type_info = self.annotate_func(*args)
        self.annotate_external_funcs()
        return func

    def annotate_func(self, *args):
        type_info = {'i' : 'int'}
        arg_names = inspect.getargspec(self.func)[0]
        arg_names.remove('i')
        for arg, name in zip(args, arg_names):
            arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type

        self.func = annotate(self.func, **type_info)
        return self.func, type_info

    def get_type(self, type_str):
        kind, address_space, ctype, shape = get_declare_info(type_str)
        if kind == 'matrix':
            ctype = '%sp' % ctype
        return ctype

    def annotate_external_funcs(self):
        src = dedent('\n'.join(inspect.getsourcelines(self.func)[0]))
        self._src = src.splitlines()
        code = ast.parse(src)
        self.visit(code)

        mod = importlib.import_module(self.func.__module__)
        calls = filter_calls(self.calls)
        undefined = []
        for call in calls:
            f = getattr(mod, call, None)
            if f is None:
                undefined.append(call)
            elif not isinstance(f, Extern):
                f_arg_names = inspect.getargspec(f)[0]
                annotations = dict(zip(f_arg_names, self.arg_types[call]))
                new_f = annotate(f, **annotations)
                setattr(mod, call, new_f)
        if undefined:
            msg = 'The following functions are not defined:\n %s ' % (
                ', '.join(undefined)
            )
            raise NameError(msg)

    def error(self, message, node):
        msg = '\nError in code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' '*node.col_offset + '^' + '\n\n'
        msg += message
        raise NotImplementedError(msg)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and \
                node.func.id not in BUILTINS:
            self.calls.add(node.func.id)
            arg_types = []
            for arg in node.args:
                arg_type = self.visit(arg)
                arg_types.append(arg_type)
            self.arg_types[node.func.id] = arg_types

    def visit_Subscript(self, node):
        base_type = self.visit(node.value)
        if base_type.startswith('g'):
            base_type = base_type[1:]
        return base_type[:-1]

    def visit_Name(self, node):
        node_type = self.type_info.get(node.id, 'double')
        return node_type

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            self.error("Assignments can have only one target.", node)
        left, right = node.targets[0], node.value
        if isinstance(right, ast.Call) and \
           isinstance(right.func, ast.Name) and right.func.id == 'declare':
            if not isinstance(right.args[0], ast.Str):
                self.error("Argument to declare should be a string.", node)
            type = right.args[0].s
            if isinstance(left, ast.Name):
                if left.id in self.type_info:
                    self.error("Redeclaring variable not allowed")
                self.type_info[left.id] = self.get_type(type)
            elif isinstance(left, ast.Tuple):
                names = [x.id for x in left.elts]
                for name in names:
                    self.type_info[name] = self.get_type(type)


class ElementwiseJIT(Elementwise):
    def __init__(self, func, backend='cython'):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.name = func.__name__
        self.func = func
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None

    @memoize_method
    def _generate_kernel(self, *args):
        # FIXME: Memoization doesn't work with np.ndarray as argument
        self.helper = JITHelper(self.func)
        self.func = self.helper.annotate(*args)

        return self._generate()

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x, dtype=np.float64)

    def __call__(self, *args, **kw):
        c_func = self._generate_kernel(*args)
        c_args = [self._massage_arg(x) for x in args]

        if self.backend == 'cython':
            size = len(c_args[0])
            c_args.insert(0, size)
            c_func(*c_args, **kw)
        elif self.backend == 'opencl':
            c_func(*c_args, **kw)
            self.queue.finish()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            c_func(*c_args, **kw)
            event.record()
            event.synchronize()
