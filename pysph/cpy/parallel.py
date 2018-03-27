"""
TODO:

- Support for OpenCL.
- Support for known types and pluggable detect_types.
- Decorator.
- Function annotation.
- Cleanup.
"""


from mako.template import Template

from pyzoltan.core.carray import BaseArray

from pysph.base.config import get_config

from .cython_generator import get_parallel_range
from .transpiler import Transpiler


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
        self.tp.add(func)
        self.name = func.__name__
        self.func = func
        self._config = get_config()

        py_data, c_data = self.tp._cgen.get_func_signature(func)
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

    def _massage_arg(self, x):
        if isinstance(x, BaseArray):
            return x.get_npy_array()
        else:
            return x

    def __call__(self, *args):
        c_args = [self._massage_arg(x) for x in args]
        size = len(c_args[0])
        c_args.insert(0, size)
        self.c_func(*c_args)
