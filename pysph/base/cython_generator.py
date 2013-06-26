"""A simple code generator that generates high-performance Cython code
from equivalent Python code.

Note that this is not a general purpose code generator but one highly tailored
for use in PySPH for general use cases, Cython itself does a terrific job.
"""

import inspect
import logging
from mako.template import Template
from textwrap import dedent

from pysph.sph.ast_utils import get_symbols

logger = logging.getLogger(__name__)

class CythonClassHelper(object):
    def __init__(self, name='', public_vars=None, methods=None):
        self.name = name
        self.public_vars = public_vars
        self.methods = methods if methods is not None else []

    def generate(self):
        template = dedent("""
cdef class ${class_name}:
    %for name, type in public_vars.iteritems():
    cdef public ${type} ${name}
    %endfor
    def __init__(self, object obj):
        for key in obj.__dict__:
            setattr(self, key, getattr(obj, key))
%for defn, body in methods:
    ${defn}
    %for line in body.splitlines():
${line}
    %endfor
%endfor
        """)
        t = Template(text=template)
        return t.render(class_name=self.name,
                        public_vars=self.public_vars,
                        methods=self.methods)

def get_func_definition(sourcelines):
    """Given a block of source lines for a method or function,
    get the lines for the function block.
    """
    # For now return the line after the first.
    count = 1
    for line in sourcelines:
        if line.rstrip().endswith('):'):
            break
        count += 1
    return sourcelines[:count], sourcelines[count:]


def all_numeric(seq):
    """Return true if all values in given sequence are numeric.
    """
    return all(type(x) in [int, float, long] for x in seq)

class CodeGenerationError(Exception):
    pass

class CythonGenerator(object):
    def __init__(self):
        self.code = ''

    def parse(self, cls):
        name = cls.__name__
        public_vars = self._get_public_vars(cls)
        methods = self._get_methods(cls)
        helper = CythonClassHelper(name=name, public_vars=public_vars,
                                   methods=methods)
        self.code = helper.generate()

    def _get_public_vars(self, cls):
        obj = cls()
        # For now get it all from the dict.
        data = obj.__dict__
        vars = dict((name, self.detect_type(name, data[name]))
                        for name in sorted(data.keys()))
        return vars

    def _get_methods(self, cls):
        methods = []
        for name in dir(cls):
            if name.startswith('_'):
                continue
            meth = getattr(cls, name)
            if callable(meth):
                sourcelines = inspect.getsourcelines(meth)[0]
                defn, lines = get_func_definition(sourcelines)
                defn = self._get_method_spec(meth)
                body = self._get_method_body(meth, lines)
                methods.append((defn, body))
        return methods

    def _get_method_body(self, meth, lines):
        args = set(inspect.getargspec(meth).args)
        body = ''.join(lines)
        dedented_body = dedent(body)
        symbols = get_symbols(dedented_body)
        undefined = symbols - args
        declare = [' '*8 +'cdef double %s\n'%x for x in undefined]
        code = ''.join(declare) + body
        return code

    def _get_method_spec(self, meth):
        name = meth.__name__
        argspec = inspect.getargspec(meth)
        args = argspec.args
        if args and args[0] == 'self':
            args = args[1:]
        defaults = argspec.defaults
        if defaults is None or (len(args) != len(argspec.defaults)):
            msg = 'Error in method {name}, not enough default '\
                  'arguments specified.'.format(name=name)
            raise CodeGenerationError(msg)

        new_args = ['self']
        for i, arg in enumerate(args):
            value = defaults[i]
            type = self.detect_type(arg, value)
	    new_args.append('{type} {arg}'.format(type=type, arg=arg))

        arg_def = ', '.join(new_args)
        defn = 'cdef inline {name}({arg_def}):'.format(name=name,
                                                       arg_def=arg_def)
        return defn

    def get_code(self):
        return self.code

    def detect_type(self, name, value):
        """Given the variable name and value, detect its type.
        """
        if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
            return 'double*'
        if isinstance(value, int):
            return 'long'
        elif isinstance(value, basestring):
            return 'basestring'
        elif isinstance(value, float):
            return 'double'
        elif isinstance(value, (list, tuple)):
            if all_numeric(value):
                # We don't deal with integer lists for now.
                return 'double[{size}]'.format(size=len(value))
            else:
                return 'list' if isinstance(value, list) else 'tuple'
        else:
            msg = 'Sorry "{name}" is {value} ({type}) which is not implemented.'\
                    '  Treating as object.'\
                    .format(name=name, value=value, type=type(value))
            logger.warn(msg)
            return 'object'
