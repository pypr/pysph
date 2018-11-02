import ast
import numpy as np


def declare(type, num=1):
    """Declare the variable to be of the given type.

    The additional optional argument num is the number of items to return.

    Normally, the declare function only defines a variable when compiled,
    however, this function here is a pure Python implementation so that the
    same code can be executed in Python.

    Parameters
    ----------

    type: str: String representing the type.
    num: int: the number of values to return

    Examples
    --------

    >>> declare('int')
    0
    >>> declare('int', 3)
    0, 0, 0
    """
    if num == 1:
        return _declare(type)
    else:
        return tuple(_declare(type) for i in range(num))


def get_declare_info(arg):
    """Given the first argument to the declare function, return the
    (kind, address_space, type, shape), information.

    kind: is a string, 'primitive' or 'matrix'
    address_space: is the address space string.
    type: is the c data type to use.
    shape: is a tuple with the shape of the matrix.  It is None for primitives.
    """
    address_space = ''
    shape = None
    if arg.startswith(('LOCAL_MEM', 'GLOBAL_MEM')):
        idx = arg.index(' ')
        address_space = arg[:idx]
        arg = arg[idx + 1:]
    if arg.startswith('matrix'):
        kind = 'matrix'
        m_arg = ast.literal_eval(arg[7:-1])
        if isinstance(m_arg, tuple) and \
                        len(m_arg) > 1 and \
                isinstance(m_arg[1], str):
            shape = m_arg[0]
            type = m_arg[1]
        else:
            shape = m_arg
            type = 'double'
    else:
        kind = 'primitive'
        type = arg

    return kind, address_space, type, shape


def _declare(arg):
    kind, address_space, ctype, shape = get_declare_info(arg)
    if kind == 'matrix':
        dtype = C_NP_TYPE_MAP[ctype]
        return np.zeros(shape, dtype=dtype)
    else:
        if ctype in ['double', 'float']:
            return 0.0
        else:
            return 0


class Undefined(object):
    pass


class KnownType(object):
    """Simple object to specify a known type as a string.

    Smells but is convenient as the type may be one available only inside
    Cython without a corresponding Python type.
    """

    def __init__(self, type_str, base_type=''):
        """Constructor

        The ``base_type`` argument is optional and used to represent the base
        type, i.e. the type_str may be 'Foo*' but the base type will be 'Foo'
        if specified.

        Parameters
        ----------
        type_str: str: A string representation of how the type is declared.
        base_type: str: The base type of this entity. (optional)

        """
        self.type = type_str
        self.base_type = base_type

    def __repr__(self):
        if self.base_type:
            return 'KnownType("%s", "%s")' % (self.type, self.base_type)
        else:
            return 'KnownType("%s")' % self.type

    def __eq__(self, other):
        return self.type == other.type and self.base_type == other.base_type


TYPES = dict(
    float=KnownType('float'),
    double=KnownType('double'),
    int=KnownType('int'),
    long=KnownType('long'),
    uint=KnownType('unsigned int'),
    ulong=KnownType('unsigned long'),

    floatp=KnownType('float*', 'float'),
    doublep=KnownType('double*', 'double'),
    intp=KnownType('int*', 'int'),
    longp=KnownType('long*', 'long'),
    uintp=KnownType('unsigned int*', 'unsigned int'),
    ulongp=KnownType('unsigned long*', 'unsigned long'),

    gfloatp=KnownType('GLOBAL_MEM float*', 'float'),
    gdoublep=KnownType('GLOBAL_MEM double*', 'double'),
    gintp=KnownType('GLOBAL_MEM int*', 'int'),
    glongp=KnownType('GLOBAL_MEM long*', 'long'),
    guintp=KnownType('GLOBAL_MEM unsigned int*', 'unsigned int'),
    gulongp=KnownType('GLOBAL_MEM unsigned long*', 'unsigned long'),

    lfloatp=KnownType('LOCAL_MEM float*', 'float'),
    ldoublep=KnownType('LOCAL_MEM double*', 'double'),
    lintp=KnownType('LOCAL_MEM int*', 'int'),
    llongp=KnownType('LOCAL_MEM long*', 'long'),
    luintp=KnownType('LOCAL_MEM unsigned int*', 'unsigned int'),
    lulongp=KnownType('LOCAL_MEM unsigned long*', 'unsigned long'),
)


def _inject_types_in_module():
    g = globals()
    for name, type in TYPES.items():
        if name in ['int', 'long', 'float']:
            name = name + '_'
        g[name] = type


# A convenience so users can import types directly from the module.
_inject_types_in_module()

NP_C_TYPE_MAP = {
    np.dtype(np.bool): 'char',
    np.dtype(np.float32): 'float', np.dtype(np.float64): 'double',
    np.dtype(np.int8): 'char', np.dtype(np.uint8): 'unsigned char',
    np.dtype(np.int16): 'short', np.dtype(np.uint16): 'unsigned short',
    np.dtype(np.int32): 'int', np.dtype(np.uint32): 'unsigned int',
    np.dtype(np.int64): 'long', np.dtype(np.uint64): 'unsigned long'
}

C_NP_TYPE_MAP = {
    'bool': np.bool,
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


def dtype_to_ctype(dtype):
    try:
        # FIXME: pyopencl depency

        from pyopencl.compyte.dtypes import \
            dtype_to_ctype as dtype_to_ctype_pyopencl
        ctype = dtype_to_ctype_pyopencl(dtype)
    except ValueError:
        pass
    except ImportError:
        pass
    else:
        return ctype
    dtype = np.dtype(dtype)
    return NP_C_TYPE_MAP[dtype]


def ctype_to_dtype(ctype):
    return np.dtype(C_NP_TYPE_MAP[ctype])


def dtype_to_knowntype(dtype, address='scalar'):
    ctype = dtype_to_ctype(dtype)
    if 'unsigned' in ctype:
        ctype = 'u%s' % ctype.replace('unsigned ', '')
    knowntype = ctype
    if address == 'ptr':
        knowntype = '%sp' % knowntype
    elif address == 'global':
        knowntype = 'g%sp' % knowntype
    elif address == 'local':
        knowntype = 'l%sp' % knowntype
    elif address != 'scalar':
        raise ValueError("address can only be scalar,"
                         " ptr, global or local")

    return knowntype
    if knowntype in TYPES:
        return knowntype
    else:
        raise TypeError("Not a vaild KnownType")


def annotate(func=None, **kw):
    """A decorator to specify the types of a function. These types are injected
    into the functions, `__annotations__` attribute.

    An example describes this best:

    @annotate(i='int', x='floatp', return_='float')
    def f(i, x):
        return x[i]*2.0

    One could also do:

    @annotate(i='int', floatp='x, y', return_='float')
    def f(i, x, y):
       return x[i]*y[i]

    """
    data = {}

    if not kw:
        def wrapper(func):
            func.is_jit = True
            return func
    else:
        for name, type in kw.items():
            if isinstance(type, str) and ',' in type:
                for x in type.split(','):
                    data[_clean_name(x.strip())] = _get_type(name)
            else:
                data[_clean_name(name)] = _get_type(type)

        def wrapper(func):
            func.__annotations__ = data
            return func

    if func is None:
        return wrapper
    else:
        return wrapper(func)


def _clean_name(name):
    return 'return' if name == 'return_' else name


def _get_type(type):
    if isinstance(type, KnownType):
        return type
    elif type in TYPES:
        return TYPES[type]
    else:
        msg = ('Unknown type {type}, not a KnownType and not one of '
               'the pre-declared types.'.format(type=str(type)))
        raise TypeError(msg)
