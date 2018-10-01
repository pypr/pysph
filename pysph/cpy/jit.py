from functools import wraps
from textwrap import dedent, wrap

from mako.template import Template
import numpy as np
import inspect
import ast
import importlib
import warnings

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import (Transpiler, convert_to_float_if_needed,
        filter_calls, CY_BUILTIN_SYMBOLS, OCL_BUILTIN_SYMBOLS, BUILTINS)
from .types import dtype_to_ctype, annotate, get_declare_info, \
        dtype_to_knowntype
from .parallel import Elementwise, Reduction, Scan
from .extern import Extern
from .ast_utils import get_unknown_names_and_calls

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


def get_binop_return_type(a, b):
    preference_order = ['short', 'long', 'int', 'float', 'double']
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
    def __init__(self, func):
        self.func = func
        self.calls = set()
        self.arg_types = {}

    def get_type(self, type_str):
        kind, address_space, ctype, shape = get_declare_info(type_str)
        if 'unsigned' in ctype:
            ctype = ctype.replace('unsigned ', 'u')
        if kind == 'matrix':
            ctype = '%sp' % ctype
        return ctype

    def get_type_info_for_external_funcs(self):
        self.type_info = getattr(self.func, 'type_info')
        src = dedent('\n'.join(inspect.getsourcelines(self.func)[0]))
        self._src = src.splitlines()
        code = ast.parse(src)
        self.visit(code)
        return self.arg_types

    def get_return_type(self, type_info):
        self.type_info = type_info
        src = dedent('\n'.join(inspect.getsourcelines(self.func)[0]))
        self._src = src.splitlines()
        code = ast.parse(src)
        self.visit(code)
        return_type = self.type_info.get('return_', None)
        self.type_info = {}
        return return_type

    def error(self, message, node):
        msg = '\nError in code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' '*node.col_offset + '^' + '\n\n'
        msg += message
        raise NotImplementedError(msg)

    def warn(self, message, node):
        msg = '\nIn code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' '*node.col_offset + '^' + '\n\n'
        msg += message
        warnings.warn(msg)

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
                self.type_info[left.id] = self.get_type(type)
            elif isinstance(left, ast.Tuple):
                names = [x.id for x in left.elts]
                for name in names:
                    self.type_info[name] = self.get_type(type)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return self.visit(node.left)
        else:
            if isinstance(node.left, ast.Call) or \
                    isinstance(node.right, ast.Call):
                return False
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
            self.type_info['return_'] = self.visit(node.value)
        elif isinstance(node.value, ast.BinOp):
            result_type = self.visit(node.value)
            if result_type:
                self.type_info['return_'] = self.visit(node.value)
            else:
                self.warn("Return value should be a variable, subscript "\
                        "or a number. Return value will default to 'double' "\
                        "otherwise", node)
                self.type_info['return_'] = 'double'
        else:
            self.warn("Return value should be a variable, subscript "\
                    "or a number. Return value will default to 'double' "\
                    "otherwise", node)
            self.type_info['return_'] = 'double'


def get_and_annotate_external_symbols_and_calls(func, backend):
    '''Given a function, return a dictionary of all external names (with their
    values), a set of implicitly defined names, a list of functions that it
    calls ignoring standard math functions and a few other standard ones, and a
    list of Extern instances.

    If a function is not defined it will raise a ``NameError``.

    Parameters
    ----------

    func: Function to look at.
    backend: str: The backend being used.

    Returns
    -------

    names, implicits, functions, externs

    '''
    if backend == 'cython':
        ignore = CY_BUILTIN_SYMBOLS
    else:
        ignore = OCL_BUILTIN_SYMBOLS

    src = dedent('\n'.join(inspect.getsourcelines(func)[0]))
    names, calls = get_unknown_names_and_calls(src)
    names -= ignore
    calls = filter_calls(calls)
    mod = importlib.import_module(func.__module__)
    symbols = {}
    implicit = set()
    externs = []
    for name in names:
        if hasattr(mod, name):
            value = getattr(mod, name)
            if isinstance(value, Extern):
                externs.append(value)
            else:
                symbols[name] = value
        else:
            implicit.add(name)

    funcs = []
    undefined = []
    helper = AnnotationHelper(func)
    arg_types = helper.get_type_info_for_external_funcs()
    for call in calls:
        f = getattr(mod, call, None)
        if f is None:
            undefined.append(call)
        elif isinstance(f, Extern):
            externs.append(f)
        else:
            f_arg_names = inspect.getargspec(f)[0]
            annotations = dict(zip(f_arg_names, arg_types[call]))
            new_f = annotate(f, **annotations)
            funcs.append(new_f)
    if undefined:
        msg = 'The following functions are not defined:\n %s ' % (
            ', '.join(undefined)
        )
        raise NameError(msg)

    return symbols, implicit, funcs, externs


class TranspilerJIT(Transpiler):
    def __init__(self, backend='cython', incl_cluda=True):
        """Constructor.

        Parameters
        ----------

        backend: str: Backend to use.
            Can be one of 'cython', 'opencl', 'cuda' or 'python'
        """
        Transpiler.__init__(self, backend=backend, incl_cluda=incl_cluda)

    def _handle_external(self, func):
        syms, implicit, calls, externs = \
                get_and_annotate_external_symbols_and_calls(func,
                                                            self.backend)
        if implicit:
            msg = ('Warning: the following symbols are implicitly defined.\n'
                   '  %s\n'
                   'You may want to explicitly declare/define them.')
            print(msg)

        self._handle_externs(externs)
        self._handle_symbols(syms)

        for f in calls:
            self.add(f)


class ElementwiseJIT(Elementwise):
    def __init__(self, func, backend='cython'):
        backend = array.get_backend(backend)
        self.tp = TranspilerJIT(backend=backend)
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
            helper = AnnotationHelper(self.func)
            return_type = helper.get_return_type(annotations)
            if return_type:
                annotations['return_'] = return_type
            self.func = annotate(self.func, **annotations)
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
            helper = AnnotationHelper(self.func)
            return_type = helper.get_return_type(annotations)
            if return_type:
                annotations['return_'] = return_type
            self.func = annotate(self.func, **annotations)
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


class ScanJIT(Scan):
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
        self.builtin_types = {'i' : 'int', 'N' : 'int'}
        for sym in builtin_symbols:
            self.builtin_types[sym] = dtype_to_knowntype(self.dtype)

    def get_type_info_from_kwargs(self, func, **kwargs):
        type_info = {}
        arg_names = inspect.getargspec(func)[0]
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

    @memoize
    def _generate_kernel(self, **kwargs):
        if self.input_func is not None:
            annotations = self.get_type_info_from_kwargs(self.input_func,
                                                         **kwargs)
            helper = AnnotationHelper(self.input_func)
            return_type = helper.get_return_type(annotations)
            if return_type:
                annotations['return_'] = return_type
            self.input_func = annotate(self.input_func, **annotations)

        if self.output_func is not None:
            annotations = self.get_type_info_from_kwargs(self.output_func,
                                                         **kwargs)
            helper = AnnotationHelper(self.output_func)
            return_type = helper.get_return_type(annotations)
            if return_type:
                annotations['return_'] = return_type
            self.output_func = annotate(self.output_func, **annotations)

        if self.is_segment_func is not None:
            annotations = self.get_type_info_from_kwargs(self.is_segment_func,
                                                         **kwargs)
            helper = AnnotationHelper(self.is_segment_func)
            return_type = helper.get_return_type(annotations)
            if return_type:
                annotations['return_'] = return_type
            self.is_segment_func = annotate(self.is_segment_func, **annotations)
        return self._generate()

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x, dtype=np.float64)

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
