import inspect
import importlib
import math
from textwrap import dedent

from .ast_utils import get_calls
from .cython_generator import CythonGenerator
from .translator import OpenCLConverter
from .ext_module import ExtModule


BUILTINS = set(
    [x for x in dir(math) if not x.startswith('_')] +
    ['max', 'abs', 'min', 'range', 'declare']
)


def filter_calls(calls):
    '''Given a set of calls filter out the math and other builtin functions.
    '''
    return [x for x in calls if x not in BUILTINS]


def get_all_functions(func):
    '''Given a function, return a list of all functions
    that it calls ignoring standard math functions.
    '''
    src = dedent('\n'.join(inspect.getsourcelines(func)[0]))
    calls = filter_calls(get_calls(src))
    mod = importlib.import_module(func.__module__)
    return [getattr(mod, call) for call in calls]


class CodeBlock(object):
    def __init__(self, obj, code):
        self.obj = obj
        self.code = code

    def __eq__(self, other):
        if isinstance(other, CodeBlock):
            return self.obj == other.obj
        else:
            return self.obj == other


class Transpiler(object):
    def __init__(self, backend='cython'):
        """Constructor.

        Parameters
        ----------

        backend: str: Backend to use.
            Can be one of 'cython', 'opencl', or 'python'
        """
        self.backend = backend
        self.blocks = []
        self.mod = None
        if backend == 'cython':
            self._cgen = CythonGenerator()
            self.header = dedent('''
            from libc.math cimport *
            from libc.math cimport fabs as abs
            from libc.math cimport M_PI as pi
            from cython.parallel import parallel, prange
            ''')
        elif backend == 'opencl':
            self._cgen = OpenCLConverter()
            self.header = dedent('''
            #define max(x, y) fmax((double)(x), (double)(y))

            __constant double pi=M_PI;
            ''')

    def add(self, obj):
        if obj in self.blocks:
            return
        for f in get_all_functions(obj):
            self.add(f)

        if self.backend == 'cython':
            self._cgen.parse(obj)
            code = self._cgen.get_code()
        elif self.backend == 'opencl':
            code = self._cgen.parse(obj)

        cb = CodeBlock(obj, code)
        self.blocks.append(cb)

    def add_code(self, code):
        cb = CodeBlock(code, code)
        self.blocks.append(cb)

    def get_code(self):
        code = [self.header] + [x.code for x in self.blocks]
        return '\n'.join(code)

    def compile(self):
        if self.backend == 'cython':
            mod = ExtModule(self.get_code(), verbose=True)
            self.mod = mod.load()
        elif self.backend == 'opencl':
            import pyopencl as cl
            from pysph.base.opencl import get_context
            ctx = get_context()
            self.mod = cl.Program(ctx, self.get_code()).build(
                options=['-w']
            )
