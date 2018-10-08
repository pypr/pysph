from functools import wraps
from textwrap import dedent, wrap

from mako.template import Template
import numpy as np
import inspect
import ast
import importlib
import warnings
import sys
from pytools import memoize
from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import (
    Transpiler,
    convert_to_float_if_needed,
    filter_calls,
    get_external_symbols_and_calls,
    BUILTINS)
from .types import (dtype_to_ctype, get_declare_info,
                    dtype_to_knowntype, annotate)
from .extern import Extern
from .ast_utils import get_unknown_names_and_calls

import pysph.cpy.array as array
import pysph.cpy.parallel as parallel


def kernel_cache_key_args(obj, *args):
    key = [get_ctype_from_arg(arg) for arg in args]
    key.append(obj)
    return tuple(key)


def kernel_cache_key_kwargs(obj, **kwargs):
    key = [get_ctype_from_arg(arg) for arg in kwargs.values()]
    key.append(obj)
    return tuple(key)


def getargspec(f):
    getargspec_f = getattr(inspect, 'getfullargspec',
                           getattr(inspect, 'getargspec'))
    return getargspec_f(f)[0]


def get_ctype_from_arg(arg):
    if isinstance(arg, array.Array):
        return arg.gptr_type
    elif isinstance(arg, np.ndarray):
        return dtype_to_ctype(arg.dtype)
    else:
        if isinstance(arg, float):
            return 'double'
        else:
            return 'long'


def get_binop_return_type(a, b):
    if a is None or b is None:
        return None
    preference_order = ['short', 'int', 'long', 'float', 'double']
    unsigned_a = unsigned_b = False
    if a.startswith('u'):
        unsigned_a = True
        a = a[1:]
    if b.startswith('u'):
        unsigned_b = True
        b = b[1:]
    idx_a = preference_order.index(a)
    idx_b = preference_order.index(b)
    return_type = preference_order[idx_a] if idx_a > idx_b else \
        preference_order[idx_b]
    if unsigned_a and unsigned_b:
        return_type = 'u%s' % return_type
    return return_type


class AnnotationHelper(ast.NodeVisitor):
    def __init__(self, func, arg_types):
        self.func = func
        self.arg_types = arg_types
        self.var_types = arg_types.copy()
        self.external_funcs = {}
        self.warning_msg = ('''
            Function called is not marked by the annotate decorator. Argument
            type defaulting to 'double'. If the type is not 'double', store
            the value in a variable of appropriate type and use the variable
            '''
                            )

    def get_type(self, type_str):
        kind, address_space, ctype, shape = get_declare_info(type_str)
        if 'unsigned' in ctype:
            ctype = ctype.replace('unsigned ', 'u')
        if kind == 'matrix':
            ctype = '%sp' % ctype
        return ctype

    def annotate(self):
        src = dedent('\n'.join(inspect.getsourcelines(self.func)[0]))
        self._src = src.splitlines()
        code = ast.parse(src)
        self.visit(code)
        self.func = annotate(self.func, **self.arg_types)

    def error(self, message, node):
        msg = '\nError in code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' ' * node.col_offset + '^' + '\n\n'
        msg += message
        raise NotImplementedError(msg)

    def warn(self, message, node):
        msg = '\nIn code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' ' * node.col_offset + '^' + '\n\n'
        msg += message
        warnings.warn(msg)

    def visit_Call(self, node):
        # FIXME: External functions have to be at the module level
        # for this to work. Pass list of external functions to
        # make this work
        mod = importlib.import_module(self.func.__module__)
        f = getattr(mod, node.func.id, None)
        if not hasattr(f, 'is_jit'):
            return None
        if node.func.id in self.external_funcs:
            return self.external_funcs[node.func.id].arg_types.get(
                'return_', None)
        if isinstance(node.func, ast.Name) and \
                node.func.id not in BUILTINS:
            if f is None or isinstance(f, Extern):
                return None
            else:
                arg_types = []
                for arg in node.args:
                    arg_type = self.visit(arg)
                    if not arg_type:
                        self.warn(dedent(self.warning_msg), arg)
                        arg_type = 'double'
                    arg_types.append(arg_type)
                # make a new helper and call visit
                f_arg_names = getargspec(f)
                f_arg_types = dict(zip(f_arg_names, arg_types))
                f_helper = AnnotationHelper(f, f_arg_types)
                f_helper.annotate()
                self.external_funcs[node.func.id] = f_helper
                return f_helper.arg_types.get('return_', None)

    def visit_Subscript(self, node):
        base_type = self.visit(node.value)
        if base_type.startswith('g'):
            base_type = base_type[1:]
        return base_type[:-1]

    def visit_Name(self, node):
        node_type = self.var_types.get(node.id, 'double')
        return node_type

    def visit_Assign(self, node):
        # Only for declare calls
        if len(node.targets) != 1:
            self.error("Assignments can have only one target.", node)
        left, right = node.targets[0], node.value
        if isinstance(right, ast.Call) and \
           isinstance(right.func, ast.Name) and right.func.id == 'declare':
            if not isinstance(right.args[0], ast.Str):
                self.error("Argument to declare should be a string.", node)
            type = right.args[0].s
            if isinstance(left, ast.Name):
                self.var_types[left.id] = self.get_type(type)
            elif isinstance(left, ast.Tuple):
                names = [x.id for x in left.elts]
                for name in names:
                    self.var_types[name] = self.get_type(type)

    def visit_Compare(self, node):
        return 'int'

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return self.visit(node.left)
        else:
            return get_binop_return_type(self.visit(node.left),
                                         self.visit(node.right))

    def visit_Num(self, node):
        if isinstance(node.n, float):
            return_type = 'double'
        else:
            if node.n > 2147483648:
                return_type = 'long'
            else:
                return_type = 'int'
        return return_type

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name) or \
                isinstance(node.value, ast.Subscript) or \
                isinstance(node.value, ast.Num):
            self.arg_types['return_'] = self.visit(node.value)
        elif isinstance(node.value, ast.BinOp) or \
                isinstance(node.value, ast.Call):
            result_type = self.visit(node.value)
            if result_type:
                self.arg_types['return_'] = self.visit(node.value)
            else:
                self.warn(dedent(self.warning_msg), node.value)
                self.arg_types['return_'] = 'double'
        else:
            self.warn("Unknown type for return value. "
                      "Return value defaulting to 'double'", node)
            self.arg_types['return_'] = 'double'


class ElementwiseJIT(parallel.ElementwiseBase):
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
        arg_names = getargspec(self.func)
        if 'i' in arg_names:
            arg_names.remove('i')
            type_info['i'] = 'int'
        for arg, name in zip(args, arg_names):
            arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type
        return type_info

    @memoize(key=kernel_cache_key_args)
    def _generate_kernel(self, *args):
        if self.func is not None:
            arg_types = self.get_type_info_from_args(*args)
            helper = AnnotationHelper(self.func, arg_types)
            helper.annotate()
            self.func = helper.func
        return self._generate()

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

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


class ReductionJIT(parallel.ReductionBase):
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

    def get_type_info_from_args(self, *args):
        type_info = {}
        arg_names = getargspec(self.func)
        if 'i' in arg_names:
            arg_names.remove('i')
            type_info['i'] = 'int'
        for arg, name in zip(args, arg_names):
            arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type
        return type_info

    @memoize(key=kernel_cache_key_args)
    def _generate_kernel(self, *args):
        if self.func is not None:
            arg_types = self.get_type_info_from_args(*args)
            helper = AnnotationHelper(self.func, arg_types)
            helper.annotate()
            self.func = helper.func
        return self._generate()

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

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


class ScanJIT(parallel.ScanBase):
    def __init__(self, input=None, output=None, scan_expr="a+b",
                 is_segment=None, dtype=np.float64, neutral='0',
                 complex_map=False, backend='opencl'):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend, incl_cluda=False)
        self.backend = backend
        self.input_func = input
        self.output_func = output
        self.is_segment_func = is_segment
        self.complex_map = complex_map
        if input is not None:
            self.name = 'scan_' + input.__name__
        else:
            self.name = 'scan'
        self.scan_expr = scan_expr
        self.dtype = dtype
        self.type = dtype_to_ctype(dtype)
        self.arg_keys = None
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
            self.neutral = neutral
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        builtin_symbols = ['item', 'prev_item', 'last_item']
        self.builtin_types = {'i': 'int', 'N': 'int'}
        for sym in builtin_symbols:
            self.builtin_types[sym] = dtype_to_knowntype(self.dtype)

    def get_type_info_from_kwargs(self, func, **kwargs):
        type_info = {}
        arg_names = getargspec(func)
        for name in arg_names:
            arg = kwargs.get(name, None)
            if arg is None and name not in self.builtin_types:
                raise ValueError("Argument %s not found" % name)
            if name in self.builtin_types:
                arg_type = self.builtin_types[name]
            else:
                arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg_type = 'double'
            type_info[name] = arg_type
        return type_info

    @memoize(key=kernel_cache_key_kwargs, use_kwargs=True)
    def _generate_kernel(self, **kwargs):
        if self.input_func is not None:
            arg_types = self.get_type_info_from_kwargs(
                self.input_func, **kwargs)
            arg_types['return_'] = dtype_to_knowntype(self.dtype)
            helper = AnnotationHelper(self.input_func, arg_types)
            helper.annotate()
            self.input_func = helper.func

        if self.output_func is not None:
            arg_types = self.get_type_info_from_kwargs(
                self.output_func, **kwargs)
            helper = AnnotationHelper(self.output_func, arg_types)
            helper.annotate()
            self.output_func = helper.func

        if self.is_segment_func is not None:
            arg_types = self.get_type_info_from_kwargs(
                self.is_segment_func, **kwargs)
            arg_types['return_'] = 'int'
            helper = AnnotationHelper(self.is_segment_func, arg_types)
            helper.annotate()
            self.is_segment_func = helper.func

        return self._generate()

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    def __call__(self, **kwargs):
        c_func = self._generate_kernel(**kwargs)
        c_args_dict = {k: self._massage_arg(x) for k, x in kwargs.items()}

        if self.backend == 'cython':
            size = len(c_args_dict[self.arg_keys[1]])
            c_args_dict['SIZE'] = size
            c_func(*[c_args_dict[k] for k in self.arg_keys])
        elif self.backend == 'opencl':
            c_func(*[c_args_dict[k] for k in self.arg_keys])
            self.queue.finish()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            c_func(*[c_args_dict[k] for k in self.arg_keys])
            event.record()
            event.synchronize()
