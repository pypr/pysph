"""A simple code generator that generates high-performance Cython code
from equivalent Python code.

Note that this is not a general purpose code generator but one highly tailored
for use in PySPH for general use cases, Cython itself does a terrific job.
"""

import inspect
from mako.template import Template
from textwrap import dedent

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

def detect_type(name, value):
    """Given the variable name and value, detect its type.
    """
    if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return 'double*'
    if isinstance(value, int):
        return 'long'
    elif isinstance(value, float):
        return 'double'
    elif isinstance(value, (list, tuple)):
        # We don't deal with integer lists for now.
        return 'double[{size}]'.format(size=len(value))
    else:
        msg = 'Sorry "{name}" is {value} ({type}) which is not implemented'\
                .format(name=name, value=value, type=type(value))
        raise NotImplementedError(msg)

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
        vars = dict((name, detect_type(name, data[name]))
                        for name in sorted(data.keys()) )
        return vars
        
    def _get_methods(self, cls):
        methods = []
        for name in dir(cls):
            meth = getattr(cls, name)
            if callable(meth):
                sourcelines = inspect.getsourcelines(meth)[0]
                defn, body = get_func_definition(sourcelines)
                defn = self._get_method_spec(meth)
                methods.append((defn, ''.join(body)))
        return methods

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
            type = detect_type(arg, value)
            if name == '__init__':
	            new_args.append('{type} {arg}={value}'\
	               .format(type=type, arg=arg, value=value))
            else:
	            new_args.append('{type} {arg}'.format(type=type, arg=arg))

        arg_def = ', '.join(new_args)
        if name == '__init__':
            defn = 'def {name}({arg_def}):'.format(name=name, 
                                                           arg_def=arg_def)
        else:       
            defn = 'cdef inline {name}({arg_def}):'.format(name=name, 
                                                       arg_def=arg_def)
        return defn

    def get_code(self):
        return self.code