from collections import defaultdict
from mako.template import Template
from os.path import dirname, join

from pyzoltan.core import carray
from pysph.base.config import get_config
from pysph.base.cython_generator import CythonGenerator, KnownType


###############################################################################
def get_code(obj):
    """This function looks at the object and gets any additional code to
    wrap from either the `_cython_code_` method or the `_get_helpers_` method.
    """
    result = []
    if hasattr(obj, '_cython_code_'):
        code = obj._cython_code_()
        doc = '# From %s'%obj.__class__.__name__
        result.extend([doc, code] if len(code) > 0 else [])
    if hasattr(obj, '_get_helpers_'):
        cg = CythonGenerator()
        doc = '# From %s'%obj.__class__.__name__
        result.append(doc)
        for helper in obj._get_helpers_():
            cg.parse(helper)
            result.append(cg.get_code())
    return result

###############################################################################
def get_all_array_names(particle_arrays):
    """For each type of carray, find the union of the names of all particle
    array properties/constants along with their array type.  Returns a
    dictionary keyed on the name of the Array class with values being a set of
    property names for each.

    Parameters
    ----------

    particle_arrays : list
        A list of particle arrays.

    Examples
    --------

    A simple example would be::

        >>> x = np.linspace(0, 1, 10)
        >>> pa = ParticleArray(name='f', x=x)
        >>> get_all_array_names([pa])
        {'DoubleArray': {'x'}, 'IntArray': {'pid', 'tag'}, 'UIntArray': {'gid'}}
    """
    props = defaultdict(set)
    for array in particle_arrays:
        for properties in (array.properties, array.constants):
            for name, arr in properties.items():
                a_type = arr.__class__.__name__
                props[a_type].add(name)
    return dict(props)

def get_known_types_for_arrays(array_names):
    """Given all the array names from `get_all_array_names` this creates known
    types for each of them so that the code generators can use this type
    information when needed.  Note that known type info is generated for both
    source and destination style arrays.

    Parameters
    ----------

    array_names: dict
        A dictionary produced by `get_all_array_names`.

    Examples
    --------

    A simple example would be::

        >>> x = np.linspace(0, 1, 10)
        >>> pa = ParticleArray(name='f', x=x)
        >>> pa.remove_property('pid')
        >>> info = get_all_array_names([pa])
        >>> get_known_types_for_arrays(info)
        {'d_gid': KnownType("unsigned int*"),
         'd_tag': KnownType("int*"),
         'd_x': KnownType("double*"),
         's_gid': KnownType("unsigned int*"),
         's_tag': KnownType("int*"),
         's_x': KnownType("double*")}

    """
    result = {}
    for arr_type, arrays in array_names.items():
        c_type = getattr(carray, arr_type)().get_c_type()
        for arr in arrays:
            known_type = KnownType(c_type + '*')
            result['s_' + arr] = known_type
            result['d_' + arr] = known_type
    return result


###############################################################################
class AccelerationEvalCythonHelper(object):
    def __init__(self, acceleration_eval):
        self.object = acceleration_eval
        self.config = get_config()
        self.all_array_names = get_all_array_names(
            self.object.particle_arrays
        )
        self.known_types = get_known_types_for_arrays(
            self.all_array_names
        )

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = join(dirname(__file__), 'acceleration_eval_cython.mako')
        template = Template(filename=path)
        main = template.render(helper=self)
        return main

    def setup_compiled_module(self, module):
        # Create the compiled module.
        object = self.object
        acceleration_eval = module.AccelerationEval(
            object.kernel, object.all_group.equations,
            object.particle_arrays
        )
        object.set_compiled_object(acceleration_eval)

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_array_decl_for_wrapper(self):
        array_names = self.all_array_names
        decl = []
        for a_type in sorted(array_names.keys()):
            props = array_names[a_type]
            decl.append(
                'cdef public {a_type} {attrs}'.format(
                    a_type=a_type, attrs=', '.join(sorted(props))
                )
            )
        return '\n'.join(decl)

    def get_header(self):
        object = self.object
        headers = []
        headers.extend(get_code(object.kernel))

        # get headers from the Equations
        for equation in object.all_group.equations:
            headers.extend(get_code(equation))

        # Kernel wrappers.
        cg = CythonGenerator(known_types=self.known_types)
        cg.parse(object.kernel)
        headers.append(cg.get_code())

        # Equation wrappers.
        headers.append(object.all_group.get_equation_wrappers(
            self.known_types
        ))

        return '\n'.join(headers)

    def get_equation_defs(self):
        return self.object.all_group.get_equation_defs()

    def get_equation_init(self):
        return self.object.all_group.get_equation_init()

    def get_kernel_defs(self):
        return 'cdef public %s kernel'%(self.object.kernel.__class__.__name__)

    def get_kernel_init(self):
        object = self.object
        return 'self.kernel = %s(**kernel.__dict__)'%(object.kernel.__class__.__name__)

    def get_variable_declarations(self):
        group = self.object.all_group
        ctx = group.context
        return group.get_variable_declarations(ctx)

    def get_array_declarations(self):
        group = self.object.all_group
        src, dest = group.get_array_names()
        src.update(dest)
        return group.get_array_declarations(src, self.known_types)

    def get_dest_array_setup(self, dest_name, eqs_with_no_source, sources, real):
        src, dest_arrays = eqs_with_no_source.get_array_names()
        for g in sources.values():
            s, d = g.get_array_names()
            dest_arrays.update(d)
        lines = ['NP_DEST = self.%s.size(real=%s)'%(dest_name, real)]
        lines += ['%s = dst.%s.data'%(n, n[2:])
                  for n in sorted(dest_arrays)]
        return '\n'.join(lines)

    def get_src_array_setup(self, src_name, eq_group):
        src_arrays, dest = eq_group.get_array_names()
        lines = ['NP_SRC = self.%s.size()'%src_name]
        lines += ['%s = src.%s.data'%(n, n[2:])
                 for n in sorted(src_arrays)]
        return '\n'.join(lines)

    def get_parallel_block(self):
        if self.config.use_openmp:
            return "with nogil, parallel():"
        else:
            return "if True: # Placeholder used for OpenMP."

    def get_particle_array_names(self):
        parrays = [pa.name for pa in self.object.particle_arrays]
        return ', '.join(parrays)
