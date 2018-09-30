from functools import wraps
from textwrap import dedent, wrap

from mako.template import Template
import numpy as np
import inspect
import ast
import importlib

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed, \
        filter_calls, BUILTINS
from .types import dtype_to_ctype, annotate, get_declare_info
from .parallel import Elementwise, Reduction, Scan
from .extern import Extern

import pysph.cpy.array as array


def get_ctype_from_arg(arg):
    if isinstance(arg, array.Array):
        return arg.gptr_type
    elif isinstance(arg, np.ndarray):
        return dtype_to_ctype(arg.dtype)
    else:
        return False


def memoize(f):
    @wraps(f)
    def wrapper(obj, *args, **kwargs):
        if not hasattr(obj, 'cache'):
            obj.cache = dict()
        key = tuple([get_ctype_from_arg(arg) for arg in args])
        if key not in obj.cache:
            obj.cache[key] = f(obj, *args, **kwargs)
        return obj.cache[key]
    return wrapper


class JITHelper(ast.NodeVisitor):
    def __init__(self, func):
        self.func = func
        self.calls = set()
        self.arg_types = {}

    def annotate(self, type_info):
        self.type_info = type_info
        self.func = annotate(self.func, **type_info)
        mod = importlib.import_module(self.func.__module__)
        setattr(mod, self.func.__name__, self.func)
        self.annotate_external_funcs()

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
                helper = JITHelper(f)
                helper.annotate(annotations)
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

    def get_type_info_from_args(self, *args):
        type_info = {}
        arg_names = inspect.getargspec(self.func)[0]
        if 'i' in arg_names:
            arg_names.remove('i')
            type_info['i'] = 'int'
        for arg, name in zip(args, arg_names):
            arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type
        return type_info

    @memoize
    def _generate_kernel(self, *args):
        if self.func is not None:
            annotations = self.get_type_info_from_args(*args)
            self.helper = JITHelper(self.func)
            self.helper.annotate(annotations)
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


class ReductionJIT(Reduction):
    def __init__(self, reduce_expr, map_func=None, dtype_out=np.float64,
                 neutral='0', backend='cython'):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.func = map_func
        if map_func is not None:
            self.name = 'reduce_' + map_func.__name__
        else:
            self.name = 'reduce'
        self.reduce_expr = reduce_expr
        self.dtype_out = dtype_out
        self.type = dtype_to_ctype(dtype_out)
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
            self.neutral = neutral
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        #self._generate()

    def get_type_info_from_args(self, *args):
        type_info = {}
        arg_names = inspect.getargspec(self.func)[0]
        if 'i' in arg_names:
            arg_names.remove('i')
            type_info['i'] = 'int'
        for arg, name in zip(args, arg_names):
            arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type
        return type_info

    @memoize
    def _generate_kernel(self, *args):
        if self.func is not None:
            annotations = self.get_type_info_from_args(*args)
            self.helper = JITHelper(self.func)
            self.helper.annotate(annotations)
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
            return c_func(*c_args, **kw)
        elif self.backend == 'opencl':
            result = c_func(*c_args, **kw)
            self.queue.finish()
            return result.get()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            result = c_func(*c_args, **kw)
            event.record()
            event.synchronize()
            return result.get()

