from collections import defaultdict
from os.path import dirname, join, expanduser, realpath
from textwrap import dedent

from mako.template import Template
from cyarray import carray

from compyle.config import get_config
from compyle.cython_generator import (CythonGenerator, KnownType,
                                      get_parallel_range)
from compyle.ext_module import ExtModule, get_platform_dir


###############################################################################
def get_cython_code(obj):
    """This function looks at the object and gets any additional code to
    wrap from either the `_cython_code_` method or the `_get_helpers_` method.
    """
    result = []
    if hasattr(obj, '_cython_code_'):
        code = obj._cython_code_()
        doc = '# From %s' % obj.__class__.__name__
        result.extend([doc, code] if len(code) > 0 else [])
    return result


def get_helper_code(helpers):
    """Given a list of helpers, return the helper code suitably wrapped.
    """
    result = []
    result.append('# Helpers')
    cg = CythonGenerator()
    for helper in helpers:
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

    particle_array : list
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
        self._ext_mod = None
        self._module = None
        self._compute_group_map()

    ##########################################################################
    # Private interface.
    ##########################################################################
    def _compute_group_map(self):
        # Given all the groups, create a mapping from the group to an index of
        # sorts that can be used when adding the pre/post callback code.
        mapping = {}
        for g_idx, group in enumerate(self.object.mega_groups):
            mapping[group] = 'self.groups[%d]' % g_idx
            if group.has_subgroups:
                for sg_idx, sub_group in enumerate(group.data):
                    code = 'self.groups[{gid}].data[{sgid}]'.format(
                        gid=g_idx, sgid=sg_idx
                    )
                    mapping[sub_group] = code
        self._group_map = mapping

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
            object.particle_arrays, object.mega_groups
        )
        object.set_compiled_object(acceleration_eval)

    def compile(self, code):
        # Note, we do not add carray or particle_array as nnps_base would
        # have been rebuilt anyway if they changed.
        root = expanduser(join('~', '.pysph', 'source', get_platform_dir()))
        depends = ["pysph.base.nnps_base"]
        # Add pysph/base directory to inc_dirs for including spatial_hash.h
        # for SpatialHashNNPS
        extra_inc_dirs = [join(dirname(dirname(realpath(__file__))), 'base')]
        self._ext_mod = ExtModule(
            code, verbose=False, root=root, depends=depends,
            extra_inc_dirs=extra_inc_dirs
        )
        self._module = self._ext_mod.load()
        return self._module

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
        helpers = []
        headers = []
        headers.extend(get_cython_code(object.kernel))
        if hasattr(object.kernel, '_get_helpers_'):
            helpers.extend(object.kernel._get_helpers_())

        # get headers from the Equations
        for equation in object.all_group.equations:
            headers.extend(get_cython_code(equation))
            if hasattr(equation, '_get_helpers_'):
                for helper in equation._get_helpers_():
                    if helper not in helpers:
                        helpers.append(helper)

        headers.extend(get_helper_code(helpers))

        # Kernel wrappers.
        cg = CythonGenerator(known_types=self.known_types)
        cg.parse(object.kernel)
        headers.append(cg.get_code())

        # Equation wrappers.
        self.known_types['SPH_KERNEL'] = KnownType(
            object.kernel.__class__.__name__
        )
        headers.append(object.all_group.get_equation_wrappers(
            self.known_types
        ))

        return '\n'.join(headers)

    def get_equation_defs(self):
        return self.object.all_group.get_equation_defs()

    def get_equation_init(self):
        return self.object.all_group.get_equation_init()

    def get_kernel_defs(self):
        return 'cdef public %s kernel' % (
            self.object.kernel.__class__.__name__
        )

    def get_kernel_init(self):
        object = self.object
        return 'self.kernel = %s(**kernel.__dict__)' % (
            object.kernel.__class__.__name__
        )

    def get_variable_declarations(self):
        group = self.object.all_group
        ctx = group.context
        return group.get_variable_declarations(ctx)

    def get_array_declarations(self):
        group = self.object.all_group
        src, dest = group.get_array_names()
        src.update(dest)
        return group.get_array_declarations(src, self.known_types)

    def get_dest_array_setup(self, dest_name, eqs_with_no_source, sources,
                             group):
        src, dest_arrays = eqs_with_no_source.get_array_names()
        for g in sources.values():
            s, d = g.get_array_names()
            dest_arrays.update(d)
        if isinstance(group.start_idx, str):
            lines = ['D_START_IDX = self.%s.%s[0]' %
                     (dest_name, group.start_idx)]
        else:
            lines = ['D_START_IDX = %s' % group.start_idx]

        if group.stop_idx is None:
            lines += ['NP_DEST = self.%s.size(real=%s)' %
                      (dest_name, group.real)]
        elif isinstance(group.stop_idx, str):
            lines += ['NP_DEST = self.%s.%s[0]' %
                      (dest_name, group.stop_idx)]
        else:
            lines += ['NP_DEST = %s' % group.stop_idx]

        lines += ['%s = dst.%s.data' % (n, n[2:])
                  for n in sorted(dest_arrays)]
        return '\n'.join(lines)

    def get_src_array_setup(self, src_name, eq_group):
        src_arrays, dest = eq_group.get_array_names()
        lines = ['NP_SRC = self.%s.size()' % src_name]
        lines += ['%s = src.%s.data' % (n, n[2:])
                  for n in sorted(src_arrays)]
        return '\n'.join(lines)

    def get_parallel_block(self):
        if self.config.use_openmp:
            return "with nogil, parallel():"
        else:
            return "if True: # Placeholder used for OpenMP."

    def get_parallel_range(self, group, nogil=True):
        kwargs = {}
        if (group.stop_idx is not None) or group.start_idx:
            kwargs['schedule'] = 'dynamic'
            kwargs['chunksize'] = None
        if nogil:
            kwargs['nogil'] = True

        return get_parallel_range("D_START_IDX", "NP_DEST", **kwargs)

    def get_particle_array_names(self):
        parrays = [pa.name for pa in self.object.particle_arrays]
        return ', '.join(parrays)

    def get_condition_call(self, group):
        return self._group_map[group] + '.condition(t, dt)'

    def get_pre_call(self, group):
        return self._group_map[group] + '.pre()'

    def get_post_call(self, group):
        return self._group_map[group] + '.post()'

    def get_iteration_init(self, group):
        lines = [
            'max_iterations = %d' % group.max_iterations,
            'min_iterations = %d' % group.min_iterations,
            '_iteration_count = 1',
            'while True:'
        ]
        return '\n'.join(lines)

    def get_iteration_check(self, group):
        src = dedent('''\
            ###############################################################
            ## Break the iteration for the group.
            ###############################################################
            if ((_iteration_count >= min_iterations)
               and (%s or (_iteration_count == max_iterations))):
                _iteration_count = 1
                break
            _iteration_count += 1
        ''' % group.get_converged_condition())
        return src
