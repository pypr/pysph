from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from pysph.sph.equation import Group, get_arrays_used_in_equation


###############################################################################
def group_equations(equations):
    """Checks the given equations and ensures the following:

     - Raises an error if the user  mixes Groups and Equations.

     - If only equations are given as a list, return a single group with all
       these equations.
    """
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
        props = set(list(array.properties.keys()) +
                    list(array.constants.keys()))
        if not eq_props < props:
            errors[array.name].update(eq_props - props)

    errors = defaultdict(set)
    _check_array(p_arrays[equation.dest], eq_dest, errors)
    if equation.sources is not None:
        for src in equation.sources:
            _check_array(p_arrays[src], eq_src, errors)

    if len(errors) > 0:
        msg = "ERROR: Missing array properties for equation: %s\n"%equation.name
        for name, missing in errors.items():
            msg += "Array '%s' missing properties %s.\n"%(name, missing)
        print(msg)
        raise RuntimeError(msg)


###############################################################################
class MegaGroup(object):
    """A mega-group refactors actual equation Groups into a more
    organized form as described below.  They inherit all properties of the
    Group so these can be used while generating code but delegate the tasks to
    real groups underneath that are assembled from the original group.

    MegaGroups are organized as:
        {destination: (eqs_with_no_source, sources, all_eqs)}
        eqs_with_no_source: Group([equations]) all SPH Equations with no source.
        sources are {source: Group([equations...])}
        all_eqs is a Group of all equations having this destination.

    This is what is stored in the `data` attribute.
    """
    def __init__(self, group):
        self._orig_group = group
        self._copy_props(group)
        self.data = self._make_data(group)

    def get_converged_condition(self):
        return self._orig_group.get_converged_condition()

    def _copy_props(self, group):
        for key in ('real', 'update_nnps', 'iterate',
                    'max_iterations', 'min_iterations', 'has_subgroups'):
            setattr(self, key, getattr(group, key))

    def _make_data(self, group):
        equations = group.equations

        if group.has_subgroups:
            return [MegaGroup(g) for g in equations]

        dest_list = []
        for equation in equations:
            dest = equation.dest
            if dest not in dest_list:
                dest_list.append(dest)

        dests = OrderedDict()
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


###############################################################################
class AccelerationEval(object):
    def __init__(self, particle_arrays, equations, kernel, mode='serial'):
        """

        Parameters
        ----------

        particle_arrays: list(ParticleArray): list of particle arrays to use.
        equations: list: A list of equations/groups.
        kernel: The kernel to use.
        parallel: str: One of 'serial', 'mpi'.
        """
        self.particle_arrays = particle_arrays
        self.equation_groups = group_equations(equations)
        self.kernel = kernel
        self.nnps = None
        self.mode = mode

        all_equations = []
        for group in self.equation_groups:
            if group.has_subgroups:
                for g in group.equations:
                    all_equations.extend(g.equations)
            else:
                all_equations.extend(group.equations)
        self.all_group = Group(equations=all_equations)

        for equation in all_equations:
            check_equation_array_properties(equation, particle_arrays)

        self.mega_groups = [MegaGroup(g) for g in self.equation_groups]
        self.c_acceleration_eval = None

    ##########################################################################
    # Public interface.
    ##########################################################################
    def compute(self, t, dt):
        """Compute the accelerations given the current time, t, and the
        timestep, dt.
        """
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
