"""A set of parallel algorithms that allow users to solve a variety of
problems. These functions are heavily inspired by the same functionality
provided in pyopencl. However, we also provide Cython implementations for these
and unify the syntax in a transparent way which allows users to write the code
once and have it run on different execution backends.

"""

from functools import wraps
from textwrap import wrap

from mako.template import Template
import numpy as np

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed
from .array import Array, get_backend
from .types import dtype_to_ctype

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

cdef double INFTY = float('inf')

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

scan_cy_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
cimport openmp
cimport numpy as np
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
        a, b = b, a % b
    return a

cdef int get_stride(int sz, int itemsize):
    return sz / gcd(sz, itemsize)


cdef void c_${name}(${c_arg_sig}):
    cdef int i, n_thread, tid, stride, sz, N

    N = SIZE
    n_thread = get_number_of_threads()
    sz = sizeof(${type})

    # This striding is to do 64 bit alignment to prevent false sharing.
    stride = get_stride(64, sz)

    cdef ${type}* buffer
    buffer = <${type}*> malloc(n_thread * stride * sz)

    % if use_segment:
    cdef int* scan_seg_flags
    cdef int* chunk_new_segment
    scan_seg_flags = <int*> malloc(SIZE * sizeof(int))
    chunk_new_segment = <int*> malloc(n_thread * stride * sizeof(int))
    % endif

    % if complex_map:
    cdef ${type}* map_output
    map_output = <${type}*> malloc(SIZE * sz)
    % endif

    if buffer == NULL:
        abort()

    cdef int buffer_idx, start, end, has_segment
    cdef ${type} a, b, temp
    # This chunksize would divide input data equally
    # between threads
    # cdef int chunksize = (SIZE + n_thread - 1) // n_thread

    # A chunk of 1 MB per thread
    cdef int chunksize = 1048576 / sz
    cdef int offset = 0
    cdef ${type} global_carry = ${neutral}
    cdef ${type} last_item
    cdef ${type} carry, item, prev_item

    while offset < SIZE:
        # Pass 1
        with nogil, parallel():
            tid = threadid()
            buffer_idx = tid * stride

            start = offset + tid * chunksize
            end = offset + min((tid + 1) * chunksize, SIZE)
            has_segment = 0

            temp = ${neutral}
            for i in range(start, end):

                % if use_segment:
                # Generate segment flags
                scan_seg_flags[i] = ${is_segment_start_expr}
                if (scan_seg_flags[i]):
                    has_segment = 1
                % endif

                # Carry
                % if use_segment:
                if (scan_seg_flags[i]):
                    a = ${neutral}
                else:
                    a = temp
                % else:
                a = temp
                % endif

                # Map
                b = ${input_expr}
                % if complex_map:
                map_output[i] = b
                % endif

                # Scan
                temp = ${scan_expr}

            buffer[buffer_idx] = temp
            % if use_segment:
            chunk_new_segment[buffer_idx] = has_segment
            % endif

        # Pass 2: Aggregate chunks
        # Add previous carry to buffer[0]
        % if use_segment:
        if chunk_new_segment[0]:
            a = ${neutral}
        else:
            a = global_carry
        b = buffer[0]
        % else:
        a = global_carry
        b = buffer[0]
        % endif
        buffer[0] = ${scan_expr}

        for i in range(n_thread - 1):
            % if use_segment:

            # With segmented scan
            if chunk_new_segment[(i + 1) * stride]:
                a = ${neutral}
            else:
                a = buffer[i * stride]
            b = buffer[(i + 1) * stride]
            buffer[(i + 1) * stride] = ${scan_expr}

            % else:

            # Without segmented scan
            a = buffer[i * stride]
            b = buffer[(i + 1) * stride]
            buffer[(i + 1) * stride] = ${scan_expr}

            % endif

        last_item = buffer[(n_thread - 1) * stride]

        # Shift buffer to right by 1 unit
        for i in range(n_thread - 1, 0, -1):
            buffer[i * stride] = buffer[(i - 1) * stride]

        buffer[0] = global_carry
        global_carry = last_item

        # Pass 3: Output
        with nogil, parallel():
            tid = threadid()
            buffer_idx = tid * stride
            carry = buffer[buffer_idx]

            start = offset + tid * chunksize
            end = offset + min((tid + 1) * chunksize, SIZE)

            for i in range(start, end):
                # Output
                % if use_segment:
                if scan_seg_flags[i]:
                    a = ${neutral}
                else:
                    a = carry
                % else:
                a = carry
                % endif

                % if complex_map:
                b = map_output[i]
                % else:
                b = ${input_expr}
                % endif

                % if calc_prev_item:
                prev_item = carry
                % endif

                carry = ${scan_expr}
                item = carry

                ${output_expr}
        offset += chunksize * n_thread

    # Clean up
    free(buffer)

    % if use_segment:
    free(scan_seg_flags)
    free(chunk_new_segment)
    % endif

    % if complex_map:
    free(map_output)
    % endif

cpdef py_${name}(${py_arg_sig}):
    return c_${name}(${py_args})
'''

# No support for last_item in single thread
scan_cy_single_thread_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
cimport openmp
cimport numpy as np

cdef void c_${name}(${c_arg_sig}):
    cdef int i, N, seg_flag_here
    cdef ${type} a, b, item
    N = SIZE

    a = ${neutral}

    for i in range(N):
        # Segment operation
        % if use_segment:
        seg_flag_here = ${is_segment_start_expr}
        if seg_flag_here:
            a = ${neutral}
        % endif

        # Map
        b = ${input_expr}

        % if calc_prev_item:
        prev_item = a
        % endif

        # Scan
        a = ${scan_expr}
        item = a

        # Output
        ${output_expr}

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

            from .opencl import get_context, get_queue
            from pyopencl.elementwise import ElementwiseKernel
            from pyopencl._cluda import CLUDA_PREAMBLE
            ctx = get_context()
            self.queue = get_queue()
            name = self.func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(', '.join(c_data[0][1:]))
            preamble = convert_to_float_if_needed(self.tp.get_code())
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )
            knl = ElementwiseKernel(
                ctx,
                arguments=arguments,
                operation=expr,
                preamble="\n".join([cluda_preamble, preamble])
            )
            self.c_func = knl
        elif self.backend == 'cuda':
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            self._correct_opencl_address_space(c_data)

            from .cuda import set_context
            set_context()
            from pycuda.elementwise import ElementwiseKernel
            from pycuda._cluda import CLUDA_PREAMBLE
            name = self.func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(', '.join(c_data[0][1:]))
            preamble = convert_to_float_if_needed(self.tp.get_code())
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )
            knl = ElementwiseKernel(
                arguments=arguments,
                operation=expr,
                preamble="\n".join([cluda_preamble, preamble])
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
            if '*' in arg and 'GLOBAL_MEM' not in arg:
                return 'GLOBAL_MEM ' + arg
            else:
                return arg

        args = [_add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            'WITHIN_KERNEL void {func}({args})'.format(
                func=self.func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, Array):
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
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            self.c_func(*c_args, **kw)
            event.record()
            event.synchronize()


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
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
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

            from .opencl import get_context, get_queue
            from pyopencl.reduction import ReductionKernel
            from pyopencl._cluda import CLUDA_PREAMBLE
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )

            ctx = get_context()
            self.queue = get_queue()
            knl = ReductionKernel(
                ctx,
                dtype_out=self.dtype_out,
                neutral=self.neutral,
                reduce_expr=self.reduce_expr,
                map_expr=expr,
                arguments=arguments,
                preamble="\n".join([cluda_preamble, preamble])
            )
            self.c_func = knl
        elif self.backend == 'cuda':
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

            from .cuda import set_context
            set_context()
            from pycuda.reduction import ReductionKernel
            from pycuda._cluda import CLUDA_PREAMBLE
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )

            knl = ReductionKernel(
                dtype_out=self.dtype_out,
                neutral=self.neutral,
                reduce_expr=self.reduce_expr,
                map_expr=expr,
                arguments=arguments,
                preamble="\n".join([cluda_preamble, preamble])
            )
            self.c_func = knl

    def _correct_return_type(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        code[0] = "cdef inline {type} {name}({args}) nogil:".format(
            type=self.type, name=self.func.__name__, args=', '.join(c_data[0])
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _add_address_space(self, arg):
        if '*' in arg and 'GLOBAL_MEM' not in arg:
            return 'GLOBAL_MEM ' + arg
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
            'WITHIN_KERNEL {type} {func}({args})'.format(
                type=self.type,
                func=self.func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, Array):
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
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            result = self.c_func(*c_args)
            event.record()
            event.synchronize()
            return result.get()


class Scan(object):
    def __init__(self, input=None, output=None, scan_expr="a+b",
                 is_segment=None, dtype=np.float64, neutral='0',
                 complex_map=False,
                 backend='opencl'):
        backend = get_backend(backend)
        self.tp = Transpiler(backend=backend)
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
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
            self.neutral = neutral
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        self._generate()

    def _correct_return_type(self, c_data, modifier):
        code = self.tp.blocks[-1].code.splitlines()
        code[0] = "cdef inline {type} {name}_{modifier}({args}) nogil:".format(
            type=self.type, name=self.name, modifier=modifier,
            args=', '.join(c_data[0])
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _include_prev_item(self):
        if 'prev_item' in self.tp.blocks[-1].code:
            return True
        else:
            return False

    def _include_last_item(self):
        if 'last_item' in self.tp.blocks[-1].code:
            return True
        else:
            return False

    def _wrap_ocl_function(self, func):
        if func is not None:
            self.tp.add(func)
            py_data, c_data = self.cython_gen.get_func_signature(func)
            self._correct_opencl_address_space(c_data, func)
            name = func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(
                ', '.join(c_data[0][1:])
            )
        else:
            arguments = ''
            expr = None
        return expr, arguments

    def _ignore_arg(self, arg_name):
        if arg_name in ['item', 'prev_item', 'last_item', 'i', 'N']:
            return True
        return False

    def _num_ignore_args(self, c_data):
        result = 0
        for arg_name in c_data[1][:]:
            if self._ignore_arg(arg_name):
                result += 1
            else:
                break
        return result

    def _generate(self):
        if self.backend == 'opencl':
            input_expr, input_args = self._wrap_ocl_function(self.input_func)
            output_expr, output_args = self._wrap_ocl_function(
                self.output_func
            )
            segment_expr, segment_args = self._wrap_ocl_function(
                self.is_segment_func
            )

            preamble = convert_to_float_if_needed(self.tp.get_code())

            from .opencl import get_context, get_queue
            from pyopencl.scan import GenericScanKernel
            ctx = get_context()
            self.queue = get_queue()
            knl = GenericScanKernel(
                ctx,
                dtype=self.dtype,
                arguments=input_args,
                input_expr=input_expr,
                scan_expr=self.scan_expr,
                neutral=self.neutral,
                output_statement=output_expr,
                is_segment_start_expr=segment_expr,
                preamble=preamble
            )
            self.c_func = knl
        elif self.backend == 'cython':
            if self.input_func is not None:
                self.tp.add(self.input_func)
                py_data, c_data = \
                    self.cython_gen.get_func_signature(self.input_func)
                self._correct_return_type(c_data, 'input')
                name = self.name
                cargs = ', '.join(c_data[1])
                input_expr = '{name}_input({cargs})'.format(name=name,
                                                            cargs=cargs)
            else:
                # The first value of the arrays (int i) are all ignored
                # later while building a list of arguments
                py_data = (['int i', '{type}[:] inp'.format(type=self.type)],
                           ['i', '&inp[0]'])
                c_data = (['int i', '{type}* inp'.format(type=self.type)],
                          ['i', 'inp'])
                input_expr = 'inp[i]'

            use_segment = False
            segment_expr = ''
            if self.is_segment_func is not None:
                self.tp.add(self.is_segment_func)
                segment_py_data, segment_c_data = \
                    self.cython_gen.get_func_signature(self.is_segment_func)
                self._correct_return_type(segment_c_data, 'segment')

                use_segment = True

                cargs = ', '.join(segment_c_data[1])
                segment_expr = '{name}_segment({cargs})'.format(name=self.name,
                                                                cargs=cargs)
                n_ignore = self._num_ignore_args(segment_c_data)
                py_data = (py_data[0] + segment_py_data[0][n_ignore:],
                           py_data[1] + segment_py_data[1][n_ignore:])
                c_data = (c_data[0] + segment_c_data[0][n_ignore:],
                          c_data[1] + segment_c_data[1][n_ignore:])

            calc_last_item = False
            calc_prev_item = False

            if self.output_func is not None:
                self.tp.add(self.output_func)
                output_py_data, output_c_data = \
                    self.cython_gen.get_func_signature(self.output_func)
                self._correct_return_type(output_c_data, 'output')

                calc_last_item = self._include_last_item()
                calc_prev_item = self._include_prev_item()

                name = self.name
                cargs = ', '.join(output_c_data[1])
                output_expr = '{name}_output({cargs})'.format(name=name,
                                                              cargs=cargs)

                n_ignore = self._num_ignore_args(output_c_data)

                py_data = (py_data[0] + output_py_data[0][n_ignore:],
                           py_data[1] + output_py_data[1][n_ignore:])
                c_data = (c_data[0] + output_c_data[0][n_ignore:],
                          c_data[1] + output_c_data[1][n_ignore:])
            else:
                output_expr = ''

            py_defn = ['long SIZE'] + py_data[0][1:]
            c_defn = ['long SIZE'] + c_data[0][1:]
            py_args = ['SIZE'] + py_data[1][1:]

            if self._config.use_openmp:
                template = Template(text=scan_cy_template)
            else:
                template = Template(text=scan_cy_single_thread_template)
            src = template.render(
                name=self.name,
                type=self.type,
                input_expr=input_expr,
                scan_expr=self.scan_expr,
                output_expr=output_expr,
                neutral=self.neutral,
                c_arg_sig=', '.join(c_defn),
                py_arg_sig=', '.join(py_defn),
                py_args=', '.join(py_args),
                openmp=self._config.use_openmp,
                calc_last_item=calc_last_item,
                calc_prev_item=calc_prev_item,
                use_segment=use_segment,
                is_segment_start_expr=segment_expr,
                complex_map=self.complex_map
            )
            self.tp.add_code(src)
            self.tp.compile()
            self.c_func = getattr(self.tp.mod, 'py_' + self.name)

    def _add_address_space(self, arg):
        if '*' in arg and '__global' not in arg:
            return '__global ' + arg
        else:
            return arg

    def _correct_opencl_address_space(self, c_data, func):
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
                func=func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, Array):
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
