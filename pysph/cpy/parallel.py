"""A set of parallel algorithms that allow users to solve a variety of
problems. These functions are heavily inspired by the same functionality
provided in pyopencl. However, we also provide Cython implementations for these
and unify the syntax in a transparent way which allows users to write the code
once and have it run on different execution backends.

"""

import re
from functools import wraps
import inspect
from textwrap import wrap

from mako.template import Template
import numpy as np

from pyzoltan.core.carray import BaseArray
from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed
from .types import KnownType
from .array import Array, get_backend


NP_C_TYPE_MAP = {
    np.bool: 'bint',
    np.float32: 'float', np.float64: 'double',
    np.int8: 'char', np.uint8: 'unsigned char',
    np.int16: 'short', np.uint16: 'unsigned short',
    np.int32: 'int', np.uint32: 'unsigned int',
    np.int64: 'long', np.uint64: 'unsigned long'
}

C_NP_TYPE_MAP = {
    'bint': np.bool,
    'char': np.int8,
    'double': np.float64,
    'float': np.float32,
    'int': np.int32,
    'long': np.int64,
    'short': np.int16,
    'unsigned char': np.uint8,
    'unsigned int': np.uint32,
    'unsigned long': np.uint64,
    'unsigned short': np.uint16
}


LID_0 = LDIM_0 = GDIM_0 = GID_0 = 0


def local_barrier():
    """Dummy method to keep Python happy.

    This is a valid function in OpenCL but has no meaning in Python for now.
    """
    pass


def dtype_to_ctype(dtype):
    return NP_C_TYPE_MAP[dtype]


elementwise_cy_template = '''
from cython.parallel import parallel, prange

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

reduction_cy_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport INFINITY
cimport openmp

cpdef int get_number_of_threads():
% if openmp:
    cdef int i, n
    with nogil, parallel():
        for i in prange(1):
            n = openmp.omp_get_num_threads()
    return n
% else:
    return 1
% endif

cdef int gcd(int a, int b):
    while b != 0:
        a, b = b, a%b
    return a

cdef int get_stride(sz, itemsize):
    return sz/gcd(sz, itemsize)


cdef ${type} c_${name}(${c_arg_sig}):
    cdef int i, n_thread, tid, stride, sz
    cdef ${type} a, b
    n_thread = get_number_of_threads()
    sz = sizeof(${type})

    # This striding is to do 64 bit alignment to prevent false sharing.
    stride = get_stride(64, sz)
    cdef ${type}* buffer
    buffer = <${type}*>malloc(n_thread*stride*sz)
    if buffer == NULL:
        abort()

%if openmp:
    with nogil, parallel():
% else:
    if 1:
% endif
        tid = threadid()
        buffer[tid*stride] = ${neutral}
%if openmp:
        for i in ${get_parallel_range("SIZE")}:
%else:
        for i in range(SIZE):
%endif
            a = buffer[tid*stride]
            b = ${map_expr}
            buffer[tid*stride] = ${reduce_expr}

    a = ${neutral}
    for i in range(n_thread):
        b = buffer[i*stride]
        a = ${reduce_expr}

    free(buffer)
    return a


cpdef py_${name}(${py_arg_sig}):
    return c_${name}(${py_args})
'''


class Elementwise(object):
    def __init__(self, func, backend='cython'):
        backend = get_backend(backend)
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
            expr = '{func}({args})'.format(
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

    def __call__(self, *args, **kw):
        c_args = [self._massage_arg(x) for x in args]
        if self.backend == 'cython':
            size = len(c_args[0])
            c_args.insert(0, size)
            self.c_func(*c_args, **kw)
        elif self.backend == 'opencl':
            self.c_func(*c_args, **kw)
            self.queue.finish()


def elementwise(func=None, backend=None):
    def _wrapper(function):
        return wraps(function)(Elementwise(function, backend=backend))
    if func is None:
        return _wrapper
    else:
        return _wrapper(func)


class Reduction(object):
    def __init__(self, reduce_expr, map_func=None, dtype_out=np.float64,
                 neutral='0', backend='cython'):
        backend = get_backend(backend)
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
        self.neutral = neutral
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        self._generate()

    def _generate(self):
        if self.backend == 'cython':
            if self.func is not None:
                self.tp.add(self.func)
                py_data, c_data = self.cython_gen.get_func_signature(self.func)
                self._correct_return_type(c_data)
                name = self.func.__name__
                cargs = ', '.join(c_data[1])
                map_expr = '{name}({cargs})'.format(name=name, cargs=cargs)
            else:
                py_data = (['int i', '{type}[:] inp'.format(type=self.type)],
                           ['i', '&inp[0]'])
                c_data = (['int i', '{type}* inp'.format(type=self.type)],
                          ['i', 'inp'])
                map_expr = 'inp[i]'
            py_defn = ['long SIZE'] + py_data[0][1:]
            c_defn = ['long SIZE'] + c_data[0][1:]
            py_args = ['SIZE'] + py_data[1][1:]
            template = Template(text=reduction_cy_template)
            src = template.render(
                name=self.name,
                type=self.type,
                map_expr=map_expr,
                reduce_expr=self.reduce_expr,
                neutral=self.neutral,
                c_arg_sig=', '.join(c_defn),
                py_arg_sig=', '.join(py_defn),
                py_args=', '.join(py_args),
                openmp=self._config.use_openmp,
                get_parallel_range=get_parallel_range
            )
            self.tp.add_code(src)
            self.tp.compile()
            self.c_func = getattr(self.tp.mod, 'py_' + self.name)
        elif self.backend == 'opencl':
            if self.func is not None:
                self.tp.add(self.func)
                py_data, c_data = self.cython_gen.get_func_signature(self.func)
                self._correct_opencl_address_space(c_data)
                name = self.func.__name__
                expr = '{func}({args})'.format(
                    func=name,
                    args=', '.join(c_data[1])
                )
                arguments = convert_to_float_if_needed(
                    ', '.join(c_data[0][1:])
                )
                preamble = convert_to_float_if_needed(self.tp.get_code())
            else:
                arguments = '{type} *in'.format(type=self.type)
                expr = None
                preamble = ''

            from pysph.base.opencl import get_context, get_queue
            from pyopencl.reduction import ReductionKernel
            ctx = get_context()
            self.queue = get_queue()
            knl = ReductionKernel(
                ctx,
                dtype_out=self.dtype_out,
                neutral=self.neutral,
                reduce_expr=self.reduce_expr,
                map_expr=expr,
                arguments=arguments,
                preamble=preamble
            )
            self.c_func = knl

    def _correct_return_type(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        code[0] = "cdef inline {type} {name}({args}) nogil:".format(
            type=self.type, name=self.func.__name__, args=', '.join(c_data[0])
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _add_address_space(self, arg):
        if '*' in arg and '__global' not in arg:
            return '__global ' + arg
        else:
            return arg

    def _correct_opencl_address_space(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        header_idx = 1
        for line in code:
            if line.rstrip().endswith(')'):
                break
            header_idx += 1

        args = [self._add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            '{type} {func}({args})'.format(
                type=self.type,
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
            return self.c_func(*c_args)
        elif self.backend == 'opencl':
            result = self.c_func(*c_args)
            self.queue.finish()
            return result.get()


class LocalMem(object):
    '''A local memory specification for a GPU kernel.

    An example illustrates this best::

       >>> l = LocalMem(2)
       >>> m = l.get('double', 128)
       >>> m.size
       2048

    Note that this is basically ``sizeof(double) * 128 * 2``
    '''
    def __init__(self, size, backend=None):
        '''
        Constructor

        Parameters
        ----------

        size: int: a multiple of the current work group size.
        baackend: str: one of 'opencl', 'cuda'
        '''
        self.backend = get_backend(backend)
        if backend == 'cython':
            raise NotImplementedError(
                'LocalMem is only meaningful for the opencl/cuda backends.'
            )
        self.size = size
        self._cache = {}

    def get(self, c_type, workgroup_size):
        """Return the local memory required given the type and work group size.
        """
        key = (c_type, workgroup_size)
        if key in self._cache:
            return self._cache[key]
        elif self.backend == 'opencl':
            import pyopencl as cl
            dtype = C_NP_TYPE_MAP[c_type]
            sz = dtype().itemsize
            mem = cl.LocalMemory(sz * self.size * workgroup_size)
            self._cache[key] = mem
            return mem
        else:
            raise NotImplementedError(
                'Backend %s not implemented' % self.backend
            )


def splay(queue, n, kernel_specific_max_wg_size=None):
    dev = queue.device
    max_work_items = min(128, dev.max_work_group_size)

    if kernel_specific_max_wg_size is not None:
        max_work_items = min(max_work_items, kernel_specific_max_wg_size)

    min_work_items = min(64, max_work_items)
    full_groups = dev.max_compute_units * 4 * 8
    # 4 to overfill the device
    # 8 is an Nvidia constant--that's how many
    # groups fit onto one compute device

    if n < min_work_items:
        group_count = 1
        work_items_per_group = min_work_items
    elif n < (full_groups * min_work_items):
        group_count = (n + min_work_items - 1) // min_work_items
        work_items_per_group = min_work_items
    elif n < (full_groups * max_work_items):
        group_count = full_groups
        grp = (n + min_work_items - 1) // min_work_items
        work_items_per_group = (
            (grp + full_groups - 1) // full_groups) * min_work_items
    else:
        group_count = (n + max_work_items - 1) // max_work_items
        work_items_per_group = max_work_items

    return (group_count*work_items_per_group,), (work_items_per_group,)


class Kernel(object):
    """A simple abstraction to create GPU kernels with pure Python.

    This will not work currently with the Cython backend.

    The idea is that one can create a Python function with suitable type
    annotations along with standard names from the CLUDA header (`LDIM_0,
    LID_0, GID_0, local_barrier()`, )etc.) to write kernels in pure Python.

    Note
    ----

    This works best with functions with annotations via the @annotate decorator
    or with function annotation as we need the type information for some simple
    type checking of the passed constants.

    """
    def __init__(self, func, backend='opencl'):
        backend = get_backend(backend)
        if backend == 'cython':
            raise NotImplementedError(
                'Kernels only work with opencl/cuda backends.'
            )
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.name = func.__name__
        self.func = func
        self._config = get_config()
        self._use_double = self._config.use_double
        from pysph.base.opencl import get_queue
        self.queue = get_queue()
        self._func_info = self._get_func_info()
        self._generate()

    def _to_float(self, s):
        return re.sub(r'\bdouble\b', 'float', s)

    def _get_func_info(self):
        getfullargspec = getattr(inspect, 'getfullargspec', inspect.getargspec)
        argspec = getfullargspec(self.func)
        annotations = getattr(
            argspec, 'annotations', self.func.__annotations__
        )

        arg_info = []
        for arg in argspec.args:
            kt = annotations[arg]
            if not self._use_double:
                kt = KnownType(
                    self._to_float(kt.type), self._to_float(kt.base_type)
                )
            arg_info.append((arg, kt))
        func_info = {
            'args': arg_info,
            'return': annotations.get('return', KnownType('void'))
        }
        return func_info

    def _generate(self):
        self.tp.add(self.func)
        self._correct_opencl_address_space()

        self.tp.compile()
        self.knl = getattr(self.tp.mod, self.name)
        import pyopencl as cl
        self._max_work_group_size = self.knl.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE,
            self.queue.device
        )

    def _correct_opencl_address_space(self):
        code = self.tp.blocks[-1].code.splitlines()
        code[0] = 'KERNEL ' + code[0]
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x, type_info, workgroup_size):
        if isinstance(x, BaseArray):
            return x.get_npy_array()
        elif isinstance(x, Array):
            return x.dev.data
        elif isinstance(x, LocalMem):
            return x.get(type_info.base_type, workgroup_size)
        else:
            dtype = C_NP_TYPE_MAP[type_info.type]
            return np.array([x], dtype=dtype)

    def _get_args(self, args, workgroup_size):
        arg_info = self._func_info['args']
        c_args = []
        for arg, a_info in zip(args, arg_info):
            c_args.append(self._massage_arg(arg, a_info[1], workgroup_size))
        return c_args

    def _get_workgroup_size(self, global_size):
        gs, ls = splay(self.queue, global_size, self._max_work_group_size)
        return gs, ls

    def __call__(self, *args, **kw):
        size = args[0].data.shape
        gs = kw.pop('global_size', size)
        n = np.prod(gs)
        ls = kw.pop('local_size', None)
        if ls is not None:
            local_size = np.prod(ls)
            global_size = ((n + local_size - 1) // local_size) * local_size
            gs = (global_size, )
        else:
            gs, ls = self._get_workgroup_size(n)
        c_args = self._get_args(args, ls[0])
        prepend = [self.queue, gs, ls]
        c_args = prepend + c_args
        if self.backend == 'opencl':
            self.knl(*c_args)
            self.queue.finish()
