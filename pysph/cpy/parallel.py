"""A set of parallel algorithms that allow users to solve a variety of
problems. These functions are heavily inspired by the same functionality
provided in pyopencl. However, we also provide Cython implementations for these
and unify the syntax in a transparent way which allows users to write the code
once and have it run on different execution backends.

"""

import re
from functools import wraps
from textwrap import wrap

from mako.template import Template

from pyzoltan.core.carray import BaseArray
from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler
from .array import Array


def convert_to_float_if_needed(code):
    use_double = get_config().use_double
    if not use_double:
        code = re.sub(r'\bdouble\b', 'float', code)
    return code


elementwise_cy_template = '''
cdef c_${name}(${c_arg_sig}):
    cdef int i
%if openmp:
    with nogil, parallel():
        for i in ${get_parallel_range("SIZE")}:
%else:
    if 1:
        for i in range(SIZE):
%endif
            ${name}(${c_args})

cpdef py_${name}(${py_arg_sig}):
    c_${name}(${py_args})
'''


class Elementwise(object):
    def __init__(self, func, backend='cython'):
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.name = func.__name__
        self.func = func
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        self._generate()

    def _generate(self):
        self.tp.add(self.func)
        if self.backend == 'cython':
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            py_defn = ['long SIZE'] + py_data[0][1:]
            c_defn = ['long SIZE'] + c_data[0][1:]
            py_args = ['SIZE'] + py_data[1][1:]
            template = Template(text=elementwise_cy_template)
            src = template.render(
                name=self.name,
                c_arg_sig=', '.join(c_defn),
                c_args=', '.join(c_data[1]),
                py_arg_sig=', '.join(py_defn),
                py_args=', '.join(py_args),
                openmp=self._config.use_openmp,
                get_parallel_range=get_parallel_range
            )
            self.tp.add_code(src)
            self.tp.compile()
            self.c_func = getattr(self.tp.mod, 'py_' + self.name)
        elif self.backend == 'opencl':
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            self._correct_opencl_address_space(c_data)

            from pysph.base.opencl import get_context, get_queue
            from pyopencl.elementwise import ElementwiseKernel
            ctx = get_context()
            self.queue = get_queue()
            name = self.func.__name__
            expr = '{func}({args});'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(', '.join(c_data[0][1:]))
            preamble = convert_to_float_if_needed(self.tp.get_code())
            knl = ElementwiseKernel(
                ctx,
                arguments=arguments,
                operation=expr,
                preamble=preamble
            )
            self.c_func = knl

    def _correct_opencl_address_space(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        header_idx = 1
        for line in code:
            if line.rstrip().endswith(')'):
                break
            header_idx += 1

        def _add_address_space(arg):
            if '*' in arg and '__global' not in arg:
                return '__global ' + arg
            else:
                return arg
        args = [_add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            'void {func}({args})'.format(
                func=self.func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' '*4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, BaseArray):
            return x.get_npy_array()
        elif isinstance(x, Array):
            return x.dev
        else:
            return x

    def __call__(self, *args):
        c_args = [self._massage_arg(x) for x in args]
        if self.backend == 'cython':
            size = len(c_args[0])
            c_args.insert(0, size)
            self.c_func(*c_args)
        elif self.backend == 'opencl':
            self.c_func(*c_args)
            self.queue.finish()


def elementwise(func=None, backend=None):
    def _wrapper(function):
        return wraps(function)(Elementwise(function, backend=backend))
    if func is None:
        return _wrapper
    else:
        return _wrapper(func)
