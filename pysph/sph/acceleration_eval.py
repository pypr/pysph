from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from compyle.config import get_config
from pysph.sph.equation import (
    CUDAGroup, CythonGroup, Group, MultiStageEquations, OpenCLGroup,
    get_arrays_used_in_equation)


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
    p_arrays = dict((x.name, x) for x in particle_arrays)
    _src, _dest = get_arrays_used_in_equation(equation)
    if equation.dest not in p_arrays:
        msg = "ERROR: Equation {eq_name} has invalid dest: '{dest}'".format(
            eq_name=equation.name, dest=equation.dest
        )
        raise RuntimeError(msg)
    if not equation.no_source:
        for src in equation.sources:
            if src not in p_arrays:
                msg = "ERROR: Equation {eq_name} has invalid "\
                      "source: '{src}'".format(eq_name=equation.name, src=src)
                raise RuntimeError(msg)

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
        msg = ("ERROR: Missing array properties for equation: %s\n"
               % equation.name)
        for name, missing in errors.items():
            msg += "Array '%s' missing properties %s.\n" % (name, missing)
        print(msg)
        raise RuntimeError(msg)


def make_acceleration_evals(particle_arrays, equations, kernel,
                            mode='serial', backend=None):
    '''Returns a list of acceleration evaluators.

    If a MultiStageEquations object is given the resulting list will have
    multiple evaluators else it will have a single one.
    '''
    if isinstance(equations, MultiStageEquations):
        groups = equations.groups
    else:
        groups = [equations]
    return [
        AccelerationEval(particle_arrays, group, kernel, mode, backend)
        for group in groups
    ]


###############################################################################
class MegaGroup(object):
    """A mega-group refactors actual equation Groups into a more
    organized form as described below.  They inherit all properties of the
    Group so these can be used while generating code but delegate the tasks to
    real groups underneath that are assembled from the original group.

    MegaGroups are organized as:
        {destination: (eqs_with_no_source, sources, all_eqs)}
        eqs_with_no_source: Group([equations]) all SPH Equations with no src.
        sources are {source: Group([equations...])}
        all_eqs is a Group of all equations having this destination.

    This is what is stored in the `data` attribute.

    Note that the order of the equations in all_eqs, eqs_with_no_source and
    sources should honor the order in which the user defines them in the
    original group.
    """
    def __init__(self, group, group_cls):
        self._orig_group = group
        self.Group = group_cls
        self._copy_props(group)
        self.data = self._make_data(group)

    def get_converged_condition(self):
        return self._orig_group.get_converged_condition()

    def _copy_props(self, group):
        for key in ('real', 'update_nnps', 'iterate', 'pre', 'post',
                    'max_iterations', 'min_iterations', 'has_subgroups',
                    'condition', 'start_idx', 'stop_idx'):
            setattr(self, key, getattr(group, key))

    def _make_data(self, group):
        equations = group.equations

        if group.has_subgroups:
            return [MegaGroup(g, self.Group) for g in equations]

        dest_list = []
        for equation in equations:
            dest = equation.dest
            if dest not in dest_list:
                dest_list.append(dest)

        dests = OrderedDict()
        for dest in dest_list:
            sources = defaultdict(list)
            eqs_with_no_source = []
            all_equations = []
            for equation in equations:
                if equation.dest != dest:
                    continue
                if equation not in all_equations:
                    all_equations.append(equation)
                if equation.no_source:
                    eqs_with_no_source.append(equation)
                else:
                    for src in equation.sources:
                        sources[src].append(equation)

            for src in sources:
                eqs = sources[src]
                sources[src] = self.Group(eqs)

            dests[dest] = (self.Group(eqs_with_no_source), sources,
                           self.Group(all_equations))

        return dests


###############################################################################
class AccelerationEval(object):
    def __init__(self, particle_arrays, equations, kernel, mode='serial',
                 backend=None):
        """

        Parameters
        ----------

        particle_arrays: list(ParticleArray): list of particle arrays to use.
        equations: list: A list of equations/groups.
        kernel: The kernel to use.
        mode: str: One of 'serial', 'mpi'.
        backend: str: indicates the backend to use.
            one of ('opencl', 'cython', 'cuda', '', None)
        """
        assert backend in ('opencl', 'cython', 'cuda', '', None)
        self.backend = self._get_backend(backend)
        self.particle_arrays = particle_arrays
        self.equation_groups = group_equations(equations)
        self.kernel = kernel
        self.nnps = None
        self.mode = mode
        if self.backend == 'cython':
            self.Group = CythonGroup
        elif self.backend == 'opencl':
            self.Group = OpenCLGroup
        elif self.backend == 'cuda':
            self.Group = CUDAGroup

        all_equations = []
        for group in self.equation_groups:
            if group.has_subgroups:
                for g in group.equations:
                    all_equations.extend(g.equations)
            else:
                all_equations.extend(group.equations)
        self.all_group = self.Group(equations=all_equations)

        for equation in all_equations:
            check_equation_array_properties(equation, particle_arrays)

        self.mega_groups = [MegaGroup(g, self.Group)
                            for g in self.equation_groups]
        self.c_acceleration_eval = None

    ##########################################################################
    # Private interface.
    ##########################################################################
    def _get_backend(self, backend):
        if not backend:
            cfg = get_config()
            if cfg.use_opencl:
                backend = 'opencl'
            elif cfg.use_cuda:
                backend = 'cuda'
            else:
                backend = 'cython'
        return backend

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
