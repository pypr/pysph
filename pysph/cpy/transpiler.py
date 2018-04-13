import inspect
import importlib
import math
import re
from textwrap import dedent

from mako.template import Template

from .config import get_config
from .ast_utils import get_unknown_names_and_calls
from .cython_generator import CythonGenerator, CodeGenerationError
from .translator import OpenCLConverter
from .ext_module import ExtModule


BUILTINS = set(
    [x for x in dir(math) if not x.startswith('_')] +
    ['max', 'abs', 'min', 'range', 'declare', 'local_barrier',
     'annotate', 'printf']
)


def filter_calls(calls):
    '''Given a set of calls filter out the math and other builtin functions.
    '''
    return [x for x in calls if x not in BUILTINS]


def get_external_symbols_and_calls(func):
    '''Given a function, return a dictionary of all external names (with their
    values), a set of implicitly defined names, and list of functions that it
    calls ignoring standard math functions and a few other standard ones.

    If a function is not defined it will raise a ``NameError``.

    '''
    ignore = set('LID_0 LID_1 LID_2 GID_0 GID_1 GID_2 '
                 'LDIM_0 LDIM_1 LDIM_2 GDIM_0 GDIM_1 GDIM_2'.split())
    src = dedent('\n'.join(inspect.getsourcelines(func)[0]))
    names, calls = get_unknown_names_and_calls(src)
    names -= ignore
    calls = filter_calls(calls)
    mod = importlib.import_module(func.__module__)
    symbols = {}
    implicit = set()
    for name in names:
        if hasattr(mod, name):
            value = getattr(mod, name)
            symbols[name] = value
        else:
            implicit.add(name)

    funcs = []
    undefined = []
    for call in calls:
        f = getattr(mod, call, None)
        if f is None:
            undefined.append(call)
        else:
            funcs.append(f)
    if undefined:
        msg = 'The following functions are not defined:\n %s ' % (
            ', '.join(undefined)
        )
        raise NameError(msg)

    return symbols, implicit, funcs


def convert_to_float_if_needed(code):
    use_double = get_config().use_double
    if not use_double:
        code = re.sub(r'\bdouble\b', 'float', code)
    return code


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
        # This attribute will store the generated and compiled source for
        # debugging.
        self.source = ''
        if backend == 'cython':
            self._cgen = CythonGenerator()
            self.header = dedent('''
            from libc.stdio cimport printf
            from libc.math cimport *
            from libc.math cimport fabs as abs
            from libc.math cimport M_PI as pi
            from cython.parallel import parallel, prange
            ''')
        elif backend == 'opencl':
            from pyopencl._cluda import CLUDA_PREAMBLE
            self._cgen = OpenCLConverter()
            cluda = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )
            self.header = cluda + dedent('''
            #define max(x, y) fmax((double)(x), (double)(y))

            __constant double pi=M_PI;
            ''')

    def _handle_symbol(self, name, value):
        backend = self.backend
        value_type = type(value)
        if isinstance(value, int):
            if value > 2147483648:
                ctype = 'long'
            else:
                ctype = 'int'
        elif isinstance(value, float):
            ctype = 'double'
        elif isinstance(value, bool):
            ctype = 'bint' if backend == 'cython' else 'int'
            if backend == 'opencl':
                value = str(value).lower()
        else:
            msg = 'Unsupported type (%s) of variable "%s"' % (
                value_type, name
            )
            raise CodeGenerationError(msg)

        if self.backend == 'cython':
            return 'cdef {type} {name} {value}'.format(
                type=ctype, name=name, value=value
            )
        elif self.backend == 'opencl':
            return '#define {name} {value}'.format(
                name=name, value=value
            )

    def _handle_symbols(self, syms):
        lines = []
        comment = '# ' if self.backend == 'cython' else '// '
        if len(syms):
            hline = '{com} {line}'.format(com=comment, line='-'*70)
            code = '{com} Global constants from user namespace'.format(
                com=comment
            )
            lines.extend([hline, code, ''])
            for name, value in syms.items():
                lines.append(self._handle_symbol(name, value))
            lines.extend(['', hline])
            self.header += '\n'.join(lines)

    def _handle_external(self, func):
        syms, implicit, calls = get_external_symbols_and_calls(func)
        if implicit:
            msg = ('Warning: the following symbols are implicitly defined.\n'
                   '  %s\n'
                   'You may want to explicitly declare/define them.')
            print(msg)

        self._handle_symbols(syms)

        for f in calls:
            self.add(f)

    def add(self, obj):
        if obj in self.blocks:
            return

        self._handle_external(obj)

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
            self.source = self.get_code()
            mod = ExtModule(self.source, verbose=True)
            self.mod = mod.load()
        elif self.backend == 'opencl':
            import pyopencl as cl
            from .opencl import get_context
            ctx = get_context()
            self.source = convert_to_float_if_needed(self.get_code())
            self.mod = cl.Program(ctx, self.source).build(
                options=['-w']
            )
