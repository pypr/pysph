from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from mako.template import Template
from os.path import dirname, join

from pysph.sph.equation import Group, get_arrays_used_in_equation
from pysph.base.ext_module import ExtModule
from pysph.base.cython_generator import CythonGenerator

###############################################################################
def group_equations(equations):
    only_groups = [x for x in equations if isinstance(x, Group)]
    if len(only_groups) > 0 and len(only_groups) != len(equations):
        raise ValueError('All elements must be Groups if you use groups.')
    if len(only_groups) == 0:
        return [Group(equations)]
    else:
        return equations

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


def check_equation_array_properties(equation, particle_arrays):
    """Given an equation and the particle arrays, check if the particle arrays
    have the necessary properties.
    """
    p_arrays = {x.name:x for x in particle_arrays}
    _src, _dest = get_arrays_used_in_equation(equation)
    eq_src = set([x[2:] for x in _src])
    eq_dest = set([x[2:] for x in _dest])

    def _check_array(array, eq_props, errors):
        """Updates the `errors` with any errors.
        """
        props = set(array.properties.keys())
        if not eq_props < props:
            errors[array.name].update(eq_props - props)

    errors = defaultdict(set)
    _check_array(p_arrays[equation.dest], eq_dest, errors)
    for src in equation.sources:
        _check_array(p_arrays[src], eq_src, errors)

    if len(errors) > 0:
        msg = "ERROR: Missing array properties for equation: %s\n"%equation.name
        for name, missing in errors.iteritems():
            msg += "Array '%s' missing properties %s.\n"%(name, missing)
        print msg
        raise RuntimeError(msg)


###############################################################################
# `SPHEval` class.
###############################################################################
class SPHEval(object):
    def __init__(self, particle_arrays, equations, kernel,
                 integrator, cell_iteration=False):
        self.particle_arrays = particle_arrays
        self.equation_groups = group_equations(equations)
        self.kernel = kernel
        self.nnps = None
        self.integrator = integrator
        self.cell_iteration = cell_iteration
        for equation in equations:
            check_equation_array_properties(equation, particle_arrays)

        all_equations = []
        for group in self.equation_groups:
            all_equations.extend(group.equations)
        self.all_group = Group(equations=all_equations)

        self.groups = [self._make_group(g) for g in self.equation_groups]
        self.ext_mod = None
        self.calc = None
        self.sph_compute = None

    ##########################################################################
    # Non-public interface.
    ##########################################################################
    def _make_group(self, group):
        equations = group.equations
        dest_list = []
        for equation in equations:
            dest = equation.dest
            if dest not in dest_list:
                dest_list.append(dest)

        dests = OrderedDict()
        dests.real = group.real
        dests.update_nnps = group.update_nnps
        for dest in dest_list:
            sources = defaultdict(list)
            eqs_with_no_source = [] # For equations that have no source.
            all_eqs = set()
            for equation in equations:
                if equation.dest != dest:
                    continue
                all_eqs.add(equation)
                if equation.no_source:
                    eqs_with_no_source.append(equation)
                else:
                    for src in equation.sources:
                        sources[src].append(equation)

            for src in sources:
                eqs = sources[src]
                sources[src] = Group(eqs)

            # Sort the all_eqs set; so the order is deterministic.  Without
            # this a  user may get a recompilation for no obvious reason.
            all_equations = list(all_eqs)
            all_equations.sort(key=lambda x:x.__class__.__name__)
            dests[dest] = (Group(eqs_with_no_source), sources,
                           Group(all_equations))

        return dests

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_helpers(self):
        helpers = []
        helpers.extend(get_code(self.kernel))

        # get helpers from the Equations
        for equation in self.all_group.equations:
            helpers.extend(get_code(equation))

        # Kernel wrappers.
        cg = CythonGenerator()
        cg.parse(self.kernel)
        helpers.append(cg.get_code())

        # Equation wrappers.
        helpers.append(self.all_group.get_equation_wrappers())

        # Integrator wrappers
        helpers.append(self.integrator.get_stepper_code())

        return '\n'.join(helpers)

    def get_equation_defs(self):
        return self.all_group.get_equation_defs()

    def get_equation_init(self):
        return self.all_group.get_equation_init()

    def get_kernel_defs(self):
        return 'cdef public %s kernel'%(self.kernel.__class__.__name__)

    def get_kernel_init(self):
        return 'self.kernel = %s(**kernel.__dict__)'%(self.kernel.__class__.__name__)

    def get_variable_declarations(self):
        group = self.all_group
        ctx = group.context
        return group.get_variable_declarations(ctx)

    def get_array_declarations(self):
        group = self.all_group
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

    def get_code(self):
        helpers = self.get_helpers()
        array_names =  get_array_names(self.particle_arrays)
        parrays = [pa.name for pa in self.particle_arrays]
        pa_names = ', '.join(parrays)
        path = join(dirname(__file__), 'sph_eval.mako')
        template = Template(filename=path)
        return template.render(helpers=helpers, array_names=array_names,
                               pa_names=pa_names, object=self,
                               integrator=self.integrator)

    def set_nnps(self, nnps):
        if self.calc is None:
            self.setup()
        self.nnps = nnps
        self.calc.set_nnps(nnps)
        self.integrator.integrator.set_nnps(nnps)

    def setup(self):
        """Always call this first.
        """
        code = self.get_code()
        self.ext_mod = ExtModule(code, verbose=True)
        mod = self.ext_mod.load()
        self.calc = mod.SPHCalc(self.kernel, self.all_group.equations,
                                *self.particle_arrays)
        self.sph_compute = self.calc.compute
        integrator = mod.Integrator(self.calc, self.integrator.steppers)
        self.integrator.set_integrator(integrator)

    def compute(self, t, dt):
        self.sph_compute(t, dt)
