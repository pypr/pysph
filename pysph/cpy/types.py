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


def _declare(type):
    if type.startswith(('LOCAL_MEM', 'GLOBAL_MEM')):
        type = type[type.index(' ') + 1:]
    if type.startswith('matrix'):
        return np.zeros(eval(type[7:-1]))
    elif type in ['double', 'float']:
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


def dtype_to_ctype(dtype):
    return NP_C_TYPE_MAP[dtype]


def ctype_to_dtype(ctype):
    return C_NP_TYPE_MAP[ctype]


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
