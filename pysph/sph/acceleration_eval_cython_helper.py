from mako.template import Template
from os.path import dirname, join

from pysph.base.cython_generator import CythonGenerator


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
def get_array_names(particle_arrays):
    """Get the union of the names of all particle array properties.
    """
    props = set()
    for array in particle_arrays:
        for name in array.properties.keys():
            props.add(name)
    props.difference_update(set(('tag', 'pid', 'gid')))
    array_names = ', '.join(sorted(props))
    return array_names


###############################################################################
class AccelerationEvalCythonHelper(object):
    def __init__(self, acceleration_eval):
        self.wrapped = acceleration_eval

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        wrapped = self.wrapped
        helpers = self.get_helpers()
        array_names =  get_array_names(wrapped.particle_arrays)
        parrays = [pa.name for pa in wrapped.particle_arrays]
        pa_names = ', '.join(parrays)
        path = join(dirname(__file__), 'acceleration_eval.mako')
        template = Template(filename=path)
        main = template.render(helpers=helpers, array_names=array_names,
                               pa_names=pa_names, object=self)
        return main

    def setup_compiled_module(self, module):
        # Create the compiled module.
        wrapped = self.wrapped
        calc = module.SPHCalc(wrapped.kernel, wrapped.all_group.equations,
                            *wrapped.particle_arrays)
        wrapped.calc = calc

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_helpers(self):
        wrapped = self.wrapped
        helpers = []
        helpers.extend(get_code(wrapped.kernel))

        # get helpers from the Equations
        for equation in wrapped.all_group.equations:
            helpers.extend(get_code(equation))

        # Kernel wrappers.
        cg = CythonGenerator()
        cg.parse(wrapped.kernel)
        helpers.append(cg.get_code())

        # Equation wrappers.
        helpers.append(wrapped.all_group.get_equation_wrappers())

        return '\n'.join(helpers)

    def get_equation_defs(self):
        return self.wrapped.all_group.get_equation_defs()

    def get_equation_init(self):
        return self.wrapped.all_group.get_equation_init()

    def get_kernel_defs(self):
        return 'cdef public %s kernel'%(self.wrapped.kernel.__class__.__name__)

    def get_kernel_init(self):
        wrapped = self.wrapped
        return 'self.kernel = %s(**kernel.__dict__)'%(wrapped.kernel.__class__.__name__)

    def get_variable_declarations(self):
        group = self.wrapped.all_group
        ctx = group.context
        return group.get_variable_declarations(ctx)

    def get_array_declarations(self):
        group = self.wrapped.all_group
        src, dest = group.get_array_names()
        src.update(dest)
        return group.get_array_declarations(src)

    def get_dest_array_setup(self, dest_name, eqs_with_no_source, sources, real):
        src, dest_arrays = eqs_with_no_source.get_array_names()
        for g in sources.values():
            s, d = g.get_array_names()
            dest_arrays.update(d)
        lines = ['NP_DEST = self.%s.size(real=%s)'%(dest_name, real)]
        lines += ['%s = dst.%s.data'%(n, n[2:])
                 for n in dest_arrays]
        return '\n'.join(lines)

    def get_src_array_setup(self, src_name, eq_group):
        src_arrays, dest = eq_group.get_array_names()
        lines = ['NP_SRC = self.%s.size()'%src_name]
        lines += ['%s = src.%s.data'%(n, n[2:])
                 for n in src_arrays]
        return '\n'.join(lines)
