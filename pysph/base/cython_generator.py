"""A simple code generator that generates high-performance Cython code
from equivalent Python code.

Note that this is not a general purpose code generator but one highly tailored
for use in PySPH for general use cases, Cython itself does a terrific job.
"""

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import inspect
import logging
from mako.template import Template
from textwrap import dedent
import types


from pysph.base.ast_utils import get_assigned, has_return
from pysph.base.config import get_config


logger = logging.getLogger(__name__)

class CythonClassHelper(object):
    def __init__(self, name='', public_vars=None, methods=None):
        self.name = name
        self.public_vars = public_vars
        self.methods = methods if methods is not None else []

    def generate(self):
        template = dedent("""
cdef class ${class_name}:
    %for name, type in public_vars.items():
    cdef public ${type} ${name}
    %endfor
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

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
    try:
        types = [int, float, long]
    except NameError:
        types = [int, float]
    return all(type(x) in types for x in seq)

class CodeGenerationError(Exception):
    pass

class Undefined(object):
    pass

class KnownType(object):
    """Simple object to specify a known type as a string.

    Smells but is convenient as the type may be one available only inside
    Cython without a corresponding Python type.
    """
    def __init__(self, type_str):
        self.type = type_str

    def __repr__(self):
        return 'KnownType("%s")'%self.type


class CythonGenerator(object):
    def __init__(self, known_types=None, python_methods=False):
        """
        Parameters
        -----------

        - known_types: dict: provides default types for known arguments.

        - python_methods: bool: generate python wrapper functions.

             specifies if convenient Python friendly wrappers are to be
             generated in addition to the low-level c wrappers.
        """

        self.code = ''
        self.python_methods = python_methods
        # Methods to not wrap.
        self.ignore_methods = ['_cython_code_']
        self.known_types = known_types if known_types is not None else {}
        self._config = get_config()

    ##### Public protocol #####################################################

    def ctype_to_python(self, type_str):
        """Given a c-style type declaration obtained from the `detect_type`
        method, return a Python friendly type declaration.
        """
        return type_str.replace('*', '[:]')

    def detect_type(self, name, value):
        """Given the variable name and value, detect its type.
        """
        if isinstance(value, KnownType):
            return value.type
        if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
            return 'double*'
        if name in ['s_idx', 'd_idx']:
            return 'long'
        if value is Undefined or isinstance(value, Undefined):
            raise CodeGenerationError('Unknown type, for %s'%name)

        if isinstance(value, bool):
            return 'int'
        elif isinstance(value, int):
            return 'long'
        elif isinstance(value, str):
            return 'str'
        elif isinstance(value, float):
            return 'double'
        elif isinstance(value, (list, tuple)):
            if all_numeric(value):
                # We don't deal with integer lists for now.
                return 'double*'
            else:
                return 'list' if isinstance(value, list) else 'tuple'
        else:
            return 'object'

    def get_code(self):
        return self.code

    def parse(self, obj):
        obj_type = type(obj)
        if obj_type is types.FunctionType:
            self._parse_function(obj)
        elif hasattr(obj, '__class__'):
            self._parse_instance(obj)
        else:
            raise TypeError('Unsupport type to wrap: %s'%obj_type)

    ###### Private protocol ###################################################

    def _analyze_method(self, meth, lines):
        """Returns information about the method.

        Specifically it returns the method name, if it has a return value,
        and a list of [(arg_name, value),...].
        """
        name = meth.__name__
        body = ''.join(lines)
        returns = has_return(dedent(body))
        argspec = inspect.getargspec(meth)
        args = argspec.args
        is_method = False
        if args and args[0] == 'self':
            args = args[1:]
            is_method = True
        defaults = argspec.defaults if argspec.defaults is not None else []

        # The call_args dict is filled up with the defaults to detect
        # the appropriate type of the arguments.
        call_args = {}

        for i in range(1, len(defaults)+1):
            call_args[args[-i]] = defaults[-i]

        # Set the rest to Undefined
        for i in range(len(args) - len(defaults)):
            call_args[args[i]] = Undefined

        # Make sure any predefined quantities are suitably typed.
        call_args.update(self.known_types)

        new_args = [('self', None)] if is_method else []
        for arg in args:
            value = call_args[arg]
            new_args.append((arg, value))

        return name, returns, new_args

    def _get_c_method_spec(self, name, returns, args):
        """Returns a C definition for the method.
        """
        c_args = []
        if args and args[0][0] == 'self':
            args = args[1:]
            c_args.append('self')

        for arg, value in args:
            c_type = self.detect_type(arg, value)
            c_args.append('{type} {arg}'.format(type=c_type, arg=arg))

        c_ret = 'double' if returns else 'void'
        c_arg_def = ', '.join(c_args)
        if self._config.use_openmp:
            ignore = ['reduce', 'converged']
            gil = " nogil" if name not in ignore else ""
        else:
            gil = ""
        cdefn = 'cdef inline {ret} {name}({arg_def}){gil}:'.format(
            ret=c_ret, name=name, arg_def=c_arg_def, gil=gil
        )

        return cdefn

    def _get_methods(self, cls):
        methods = []
        for name in dir(cls):
            if name.startswith('_'):
                continue
            meth = getattr(cls, name)
            if callable(meth):
                if name in self.ignore_methods:
                    continue

                c_code, py_code = self._get_method_wrapper(meth, indent=' '*8)
                methods.append(c_code)
                if self.python_methods:
                    methods.append(py_code)

        return methods

    def _get_method_body(self, meth, lines, indent=' '*8):
        args = set(inspect.getargspec(meth).args)
        src = [self._process_body_line(line) for line in lines]
        declared = [x[0] for x in src if len(x[0]) > 0]
        cython_body = ''.join([x[1] for x in src])
        body = ''.join(lines)
        dedented_body = dedent(body)
        symbols = get_assigned(dedented_body)
        undefined = symbols - set(declared) - args
        declare = [indent +'cdef double %s\n'%x for x in sorted(undefined)]
        code = ''.join(declare) + cython_body
        return code

    def _get_method_wrapper(self, meth, indent=' '*8):
        sourcelines = inspect.getsourcelines(meth)[0]
        defn, lines = get_func_definition(sourcelines)
        m_name, returns, args = self._analyze_method(meth, lines)
        c_defn = self._get_c_method_spec(m_name, returns, args)
        c_body = self._get_method_body(meth, lines, indent=indent)
        self.code = '{defn}\n{body}'.format(defn=c_defn, body=c_body)
        if self.python_methods:
            defn, body = self._get_py_method_spec(m_name, returns, args,
                                                  indent=indent)
        else:
            defn, body = None, None
        return (c_defn, c_body), (defn, body)

    def _get_public_vars(self, obj):
        # For now get it all from the dict.
        data = obj.__dict__
        vars = OrderedDict((name, self.detect_type(name, data[name]))
                            for name in sorted(data.keys()))
        return vars

    def _get_py_method_spec(self, name, returns, args, indent=' '*8):
        """Returns a Python friendly definition for the method along with the
        wrapper function.
        """
        py_args = []
        is_method = False
        if args and args[0][0] == 'self':
            is_method = True
            args = args[1:]
            py_args.append('self')

        call_sig = []
        for arg, value in args:
            c_type = self.detect_type(arg, value)
            py_type = self.ctype_to_python(c_type)
            py_args.append('{type} {arg}'.format(type=py_type, arg=arg))
            if c_type.endswith('*'):
                call_sig.append('&{arg}[0]'.format(arg=arg))
            else:
                call_sig.append('{arg}'.format(arg=arg))

        py_ret = ' double' if returns else ''
        py_arg_def = ', '.join(py_args)
        pydefn = 'cpdef{ret} py_{name}({arg_def}):'\
                     .format(ret=py_ret, name=name, arg_def=py_arg_def)
        call = ', '.join(call_sig)
        py_ret = 'return ' if returns else ''
        py_self = 'self.' if is_method else ''
        body = indent + '{ret}{self}{name}({call})'\
                    .format(name=name, call=call, ret=py_ret, self=py_self)

        return pydefn, body

    def _handle_declare_statement(self, name, declare):
        def matrix(size):
            sz = ''.join(['[%d]'%n for n in size])
            return sz

        # Remove the "declare('" and the trailing "')".
        code = declare[9:-2]
        if code.startswith('matrix'):
            sz = matrix(eval(code[7:-1]))
            defn = 'cdef double %s%s'%(name, sz)
            return defn
        else:
            defn = 'cdef {type} {name}'.format(type=code, name=name)
            return defn

    def _parse_function(self, obj):
        c_code, py_code = self._get_method_wrapper(obj, indent=' '*4)
        code = '{defn}\n{body}'.format(defn=c_code[0], body=c_code[1])
        if self.python_methods:
            code += '\n'
            code += '{defn}\n{body}'.format(defn=py_code[0], body=py_code[1])
        self.code = code

    def _parse_instance(self, obj):
        cls = obj.__class__
        name = cls.__name__
        public_vars = self._get_public_vars(obj)
        methods = self._get_methods(cls)
        helper = CythonClassHelper(name=name, public_vars=public_vars,
                                   methods=methods)
        self.code = helper.generate()

    def _process_body_line(self, line):
        """Returns the name defined and the processed line itself.

        This hack primarily lets us declare variables from Python and inject
        them into Cython code.
        """
        if '=' in line:
            words = [x.strip() for x in line.split('=')]
            if words[1].startswith('declare'):
                name = words[0]
                declare = words[1]
                defn = self._handle_declare_statement(name, declare)
                indent = line[:line.index(name)]
                return name, indent + defn + '\n'
            else:
                return '', line
        else:
            return '', line
