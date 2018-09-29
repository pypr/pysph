from functools import wraps
from textwrap import wrap

from mako.template import Template
from pytools import memoize_method
import numpy as np
import inspect

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed
from .types import dtype_to_ctype, annotate
from .parallel import Elementwise

import pysph.cpy.array as array


def get_ctype_from_arg(arg):
    if isinstance(arg, array.Array):
        return arg.gptr_type
    elif isinstance(arg, np.ndarray):
        return dtype_to_ctype(arg.dtype)
    else:
        return False


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
        type_info = {'i' : 'int'}
        c_args = []
        arg_names = inspect.getargspec(self.func)[0]
        arg_names.remove('i')
        for arg, name in zip(args, arg_names):
            if name == 'i':
                arg_type = 'int'
            else:
                arg_type = get_ctype_from_arg(arg)
            if not arg_type:
                arg = np.asarray(arg, dtype=np.float64)
                arg_type = 'double'
            c_args.append(self._massage_arg(arg))
            type_info[name] = arg_type

        self.func = annotate(self.func, **type_info)

        return c_args, self._generate()

    def __call__(self, *args, **kw):
        c_args, c_func = self._generate_kernel(*args)

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
