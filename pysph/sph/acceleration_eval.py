from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from pysph.sph.equation import Group, get_arrays_used_in_equation


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
def check_equation_array_properties(equation, particle_arrays):
    """Given an equation and the particle arrays, check if the particle arrays
    have the necessary properties.
    """
    #p_arrays = {x.name:x for x in particle_arrays}
    p_arrays = dict((x.name, x) for x in particle_arrays)
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
    if equation.sources is not None:
        for src in equation.sources:
            _check_array(p_arrays[src], eq_src, errors)

    if len(errors) > 0:
        msg = "ERROR: Missing array properties for equation: %s\n"%equation.name
        for name, missing in errors.iteritems():
            msg += "Array '%s' missing properties %s.\n"%(name, missing)
        print msg
        raise RuntimeError(msg)


###############################################################################
class AccelerationEval(object):
    def __init__(self, particle_arrays, equations, kernel,
                 cell_iteration=False):
        self.particle_arrays = particle_arrays
        self.equation_groups = group_equations(equations)
        self.kernel = kernel
        self.nnps = None
        self.cell_iteration = cell_iteration

        all_equations = []
        for group in self.equation_groups:
            all_equations.extend(group.equations)
        self.all_group = Group(equations=all_equations)

        for equation in all_equations:
            check_equation_array_properties(equation, particle_arrays)

        self.groups = [self._make_group(g) for g in self.equation_groups]
        self.c_acceleration_eval = None

    ##########################################################################
    # Public interface.
    ##########################################################################
    def compute(self, t, dt):
        self.c_acceleration_eval.compute(t, dt)

    def set_compiled_object(self, c_acceleration_eval):
        """Set the high-performance compiled object to call internally.
        """
        self.c_acceleration_eval = c_acceleration_eval

    def set_nnps(self, nnps):
        self.nnps = nnps
        self.c_acceleration_eval.set_nnps(nnps)

    def update_particle_arrays(self, particle_arrays):
        """Call this to update the particle arrays with new ones.  Make sure
        though that the same properties exist in both or you will get a
        segfault.
        """
        self.c_acceleration_eval.update_particle_arrays(particle_arrays)

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
