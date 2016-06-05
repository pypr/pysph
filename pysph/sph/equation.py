"""Defines the basic Equation and all of its support machinery including
the Group class.
"""
# System library imports.
import ast

from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import re
from copy import deepcopy
import inspect
import itertools
import numpy
from textwrap import dedent

# Local imports.
from pysph.base.ast_utils import get_symbols
from pysph.base.cython_generator import CythonGenerator, KnownType

def camel_to_underscore(name):
    """Given a CamelCase name convert it to a name with underscores,
    i.e. camel_case.
    """
    # From stackoverflow: :P
    # http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


##############################################################################
# `Context` class.
##############################################################################
class Context(dict):
    """Based on the Bunch receipe by Alex Martelli from Active State's recipes.

    A convenience class used to specify a context in which a code block will
    execute.

    Example
    -------

    Basic usage::
        >>> c = Context(a=1, b=2)
        >>> c.a
        1
        >>> c.x = 'a'
        >>> c.x
        'a'
        >>> c.keys()
        ['a', 'x', 'b']
    """
    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError('Context has no attribute %s'%key)

    def __setattr__(self, key, value):
        self[key] = value

def get_array_names(symbols):
    """Given a set of symbols, return a set of source array names and
    a set of destination array names.
    """
    src_arrays = set(x for x in symbols
                            if x.startswith('s_') and x != 's_idx')
    dest_arrays = set(x for x in symbols
                            if x.startswith('d_') and x != 'd_idx')
    return src_arrays, dest_arrays

##############################################################################
# `BasicCodeBlock` class.
##############################################################################
class BasicCodeBlock(object):
    """Encapsulates a string of code and the context in which it executes.

    It also performs some simple analysis of the code that proves handy.
    """

    ##########################################################################
    # `object` interface.
    ##########################################################################
    def __init__(self, code, **kwargs):
        """Constructor.

        Parameters
        ----------

        code : str: source code.
        kwargs : values which define the context of the code.
        """
        self.setup(code, **kwargs)

    def __call__(self, **kwargs):
        """A simplistic test for the code that runs the code in the setup
        context with any additional arguments passed set in the context.

        Note that this will make a deepcopy of the context to prevent any
        changes to the original context.

        It returns a dictionary.

        """
        context = deepcopy(dict(self.context))
        if kwargs:
            context.update(kwargs)
        bytecode = compile(self.code, '<string>', 'exec')
        glb = globals()
        exec(bytecode, glb, context)
        return Context(**context)

    ##########################################################################
    # Private interface.
    ##########################################################################
    def _setup_context(self):
        context = self.context
        symbols = self.symbols
        for index in ('s_idx', 'd_idx'):
            if index in symbols and index not in context:
                context[index] = 0

        for a_name in itertools.chain(self.src_arrays, self.dest_arrays):
            if a_name not in context:
                context[a_name] = numpy.zeros(2, dtype=float)

    def _setup_code(self, code):
        """Perform analysis of the code and store the information in various
        attributes.
        """
        code = dedent(code)
        self.code = code
        self.ast_tree = ast.parse(code)
        self.symbols = get_symbols(self.ast_tree)

        symbols = self.symbols
        self.src_arrays, self.dest_arrays = get_array_names(symbols)
        self._setup_context()

    ##########################################################################
    # Public interface.
    ##########################################################################
    def setup(self, code, **kwargs):
        """Setup the code and context with the given arguments.

        Parameters
        ----------

        code : str: source code.

        kwargs : values which define the context of the code.
        """
        self.context = Context(**kwargs)

        if code is not None:
            self._setup_code(code)


##############################################################################
# Convenient precomputed symbols and their code.
##############################################################################
def precomputed_symbols():
    """Return a collection of predefined symbols that can be used in equations.
    """
    c = Context()
    c.HIJ = BasicCodeBlock(code="HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])", HIJ=0.0)

    c.EPS = BasicCodeBlock(code="EPS = 0.01*HIJ*HIJ", EPS=0.0)

    c.RHOIJ = BasicCodeBlock(code="RHOIJ = 0.5*(d_rho[d_idx] + s_rho[s_idx])",
                             RHOIJ=0.0)

    c.RHOIJ1 = BasicCodeBlock(code="RHOIJ1 = 1.0/RHOIJ", RHOIJ1=0.0)

    c.XIJ = BasicCodeBlock(code=dedent("""
                XIJ[0] = d_x[d_idx] - s_x[s_idx]
                XIJ[1] = d_y[d_idx] - s_y[s_idx]
                XIJ[2] = d_z[d_idx] - s_z[s_idx]
                """),
                XIJ=[0.0, 0.0, 0.0])

    c.VIJ = BasicCodeBlock(code=dedent("""
                VIJ[0] = d_u[d_idx] - s_u[s_idx]
                VIJ[1] = d_v[d_idx] - s_v[s_idx]
                VIJ[2] = d_w[d_idx] - s_w[s_idx]
                """),
                VIJ=[0.0, 0.0, 0.0])

    c.R2IJ = BasicCodeBlock(code=dedent("""
                R2IJ = XIJ[0]*XIJ[0] + XIJ[1]*XIJ[1] + XIJ[2]*XIJ[2]
                """),
                R2IJ=0.0)

    c.RIJ = BasicCodeBlock(code=dedent("""
                RIJ = sqrt(R2IJ)
                """),
                RIJ=0.0)

    c.WIJ = BasicCodeBlock(
                code="WIJ = KERNEL(XIJ, RIJ, HIJ)",
                WIJ=0.0)

    # wdeltap for tensile instability correction
    c.WDP = BasicCodeBlock(
                code="WDP = KERNEL(XIJ, DELTAP*HIJ, HIJ)",
                WDP=0.0)

    c.WI = BasicCodeBlock(
                code="WI = KERNEL(XIJ, RIJ, d_h[d_idx])",
                WI=0.0)

    c.WJ = BasicCodeBlock(
                code="WJ = KERNEL(XIJ, RIJ, s_h[s_idx])",
                WJ=0.0)

    c.DWIJ = BasicCodeBlock(
                code="GRADIENT(XIJ, RIJ, HIJ, DWIJ)",
                DWIJ=[0.0, 0.0, 0.0])

    c.DWI = BasicCodeBlock(
                code="GRADIENT(XIJ, RIJ, d_h[d_idx], DWI)",
                DWI=[0.0, 0.0, 0.0])

    c.DWJ = BasicCodeBlock(
                code="GRADIENT(XIJ, RIJ, s_h[s_idx], DWJ)",
                DWJ=[0.0, 0.0, 0.0])

    c.GHI = BasicCodeBlock(
                code="GHI = GRADH(XIJ, RIJ, d_h[d_idx])",
                GHI=0.0)

    c.GHJ = BasicCodeBlock(
                code="GHJ = GRADH(XIJ, RIJ, s_h[s_idx])",
                GHJ=0.0)

    c.GHIJ= BasicCodeBlock(
                code="GHIJ = GRADH(XIJ, RIJ, HIJ)",
                GHIJ=0.0)

    return c


def sort_precomputed(precomputed):
    """Sorts the precomputed equations in the given dictionary as per the
    dependencies of the symbols and returns an ordered dict.

    Note that this will not deal with finding any precomputed symbols that
    are dependent on other precomputed symbols.  It only sorts them in the
    right order.
    """
    weights = dict((x, None) for x in precomputed)
    pre_comp = Group.pre_comp
    # Find the dependent pre-computed symbols for each in the precomputed.
    depends = dict((x, None) for x in precomputed)
    for pre, cb in precomputed.items():
        depends[pre] = [x for x in cb.symbols if x in pre_comp and x != pre]

    # The basic algorithm is to assign weights to each of the precomputed
    # symbols based on the maximum weight of the dependencies of the
    # precomputed symbols.  This way, those with no dependencies have weight
    # zero and those with more have heigher weights. The `levels` dict stores
    # a list of precomputed symbols for each  weight.  These are then stored
    # in an ordered dict in the order of the weights to produce the output.
    levels = defaultdict(list)
    pre_comp_names = list(precomputed.keys())
    while pre_comp_names:
        for name in pre_comp_names[:]:
            wts = [weights[x] for x in depends[name]]
            if len(wts) == 0:
                weights[name] = 0
                levels[0].append(name)
                pre_comp_names.remove(name)
            elif None in wts:
                continue
            else:
                level = max(wts) + 1
                weights[name] = level
                levels[level].append(name)
                pre_comp_names.remove(name)

    result = OrderedDict()
    for level in range(len(levels)):
        for name in sorted(levels[level]):
            result[name] = pre_comp[name]

    return result


def get_predefined_types(precomp):
    """Return a dictionary that can be used by a CythonGenerator for
    the precomputed symbols.
    """
    result = {'DT_ADAPT':[0.0, 0.0, 0.0],
              'dt': 0.0,
              't': 0.0,
              'dst': KnownType('ParticleArrayWrapper'),
              'src': KnownType('ParticleArrayWrapper')}
    for sym, value in precomp.items():
        result[sym] = value.context[sym]
    return result


def get_arrays_used_in_equation(equation):
    """Return two sets, the source and destination arrays used by the equation.
    """
    src_arrays = set()
    dest_arrays = set()
    for meth_name in ('initialize', 'loop', 'post_loop'):
        meth = getattr(equation, meth_name, None)
        if meth is not None:
            args = inspect.getargspec(meth).args
            s, d = get_array_names(args)
            src_arrays.update(s)
            dest_arrays.update(d)
    return src_arrays, dest_arrays

def get_init_args(obj, method, ignore=None):
    """Return the arguments for the method given, typically an __init__.
    """
    ignore = ignore if ignore is not None else []
    spec = inspect.getargspec(method)
    keys = [k for k in spec.args[1:] if k not in ignore and k in obj.__dict__]
    args = ['%s=%r'%(k, getattr(obj, k)) for k in keys]
    return args


##############################################################################
# `Equation` class.
##############################################################################
class Equation(object):

    ##########################################################################
    # `object` interface.
    ##########################################################################
    def __init__(self, dest, sources):
        r"""
        Parameters
        ----------
        dest : str
            name of the destination particle array
        sources : list of str or None
            names of the source particle arrays
        """
        self.dest = dest
        self.sources = sources if sources is not None and len(sources) > 0 \
                                                                else None
        # Does the equation require neighbors or not.
        self.no_source = self.sources is None
        self.name = self.__class__.__name__
        # The name of the variable used in the compiled AccelerationEval
        # instance.
        self.var_name = ''

    def __repr__(self):
        name = self.__class__.__name__
        args = get_init_args(self, self.__init__, [])
        return '%s(%s)'%(name, ', '.join(args))

    def converged(self):
        """Return > 0 to indicate converged iterations and < 0 otherwise.
        """
        return 1.0

###############################################################################
# `Group` class.
###############################################################################
class Group(object):
    """A group of equations.

    This class provides some support for the code generation for the
    collection of equations.
    """

    pre_comp = precomputed_symbols()

    def __init__(self, equations, real=True, update_nnps=False, iterate=False,
                 max_iterations=1, min_iterations=0):
        """Constructor.

        Parameters
        ----------

        equations: list
            a list of equation objects.

        real: bool
            specifies if only non-remote/non-ghost particles should be
            operated on.

        update_nnps: bool
            specifies if the neighbors should be re-computed locally after
            this group

        iterate: bool
            specifies if the group should continue iterating until each
            equation's "converged()" methods returns with a positive value.

        max_iterations: int
            specifies the maximum number of times this group should be
            iterated.

        min_iterations: int
            specifies the minimum number of times this group should be
            iterated.

        Notes
        -----

        When running simulations in parallel, one should typically
        run the summation density over all particles (both local and remote)
        in each processor.  This is because we must update the
        pressure/density of the remote neighbors in the current processor.
        Otherwise the results can be incorrect with the remote particles
        having an older density.  This is also the case for the TaitEOS.  In
        these cases the group that computes the equation should set real to
        False.
        """
        self.real = real
        self.update_nnps = update_nnps

        # iterative groups
        self.iterate = iterate
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations

        only_groups = [x for x in equations if isinstance(x, Group)]
        if (len(only_groups) > 0) and (len(only_groups) != len(equations)):
            raise ValueError('All elements must be Groups if you use sub groups.')

        # This group has only sub-groups.
        self.has_subgroups = len(only_groups) > 0

        self.equations = equations
        self.src_arrays = self.dest_arrays = None

        self.update()

    ##########################################################################
    # Non-public interface.
    ##########################################################################
    def __repr__(self):
        cls = self.__class__.__name__
        eqs = [repr(eq) for eq in self.equations]
        ignore = ['equations']
        kws = ', '.join(get_init_args(self, self.__init__, ignore))
        return '%s(equations=[\n%s\n    ],\n    %s)'%(
            cls, ',\n'.join(eqs), kws
        )

    def _get_variable_decl(self, context, mode='declare'):
        decl = []
        names = list(context.keys())
        names.sort()
        for var in names:
            value = context[var]
            if isinstance(value, int):
                declare = 'cdef long ' if mode == 'declare' else ''
                decl.append('{declare}{var} = {value}'.format(declare=declare,
                                                              var=var,
                                                              value=value))
            elif isinstance(value, float):
                declare = 'cdef double ' if mode == 'declare' else ''
                decl.append('{declare}{var} = {value}'.format(declare=declare,
                                                              var=var,
                                                              value=value))
            elif isinstance(value, (list, tuple)):
                if mode == 'declare':
                    decl.append('cdef DoubleArray _{var} = '\
                        'DoubleArray(aligned({size}, 8)*self.n_threads)'.format(
                            var=var, size=len(value)
                        )
                    )
                    decl.append('cdef double* {var} = _{var}.data'\
                            .format(size=len(value), var=var))
                else:
                    pass
        return '\n'.join(decl)

    def _has_code(self, kind='loop'):
        assert kind in ('initialize', 'loop', 'post_loop', 'reduce')
        for equation in self.equations:
            if hasattr(equation, kind):
                return True

    def _get_code(self, kind='loop'):
        assert kind in ('initialize', 'loop', 'post_loop', 'reduce')
        # We assume here that precomputed quantities are only relevant
        # for loops and not post_loops and initialization.
        pre = []
        if kind == 'loop':
            for p, cb in self.precomputed.items():
                pre.append(cb.code.strip())
            if len(pre) > 0:
                pre.append('')
        code = []
        for eq in self.equations:
            meth = getattr(eq, kind, None)
            if meth is not None:
                args = inspect.getargspec(meth).args
                if 'self' in args:
                    args.remove('self')
                call_args = ', '.join(args)
                c = 'self.{eq_name}.{method}({args})'\
                      .format(eq_name=eq.var_name, method=kind, args=call_args)
                code.append(c)
        if len(code) > 0:
            code.append('')
        return '\n'.join(pre + code)

    def _set_kernel(self, code, kernel):
        if kernel is not None:
            k_func = 'self.kernel.kernel'
            g_func = 'self.kernel.gradient'
            h_func = 'self.kernel.gradient_h'
            deltap = 'self.kernel.get_deltap()'
            code = code.replace('DELTAP', deltap)
            return code.replace('GRADIENT', g_func).replace('KERNEL', k_func).replace('GRADH', h_func)
        else:
            return code

    def _setup_precomputed(self):
        """Get the precomputed symbols for this group of equations.
        """
        # Calculate the precomputed symbols for this equation.
        all_args = set()
        for equation in self.equations:
            if hasattr(equation, 'loop'):
                args = inspect.getargspec(equation.loop).args
                all_args.update(args)
        all_args.discard('self')

        pre = self.pre_comp
        precomputed = dict((s, pre[s]) for s in all_args if s in pre)

        # Now find the precomputed symbols in the pre-computed symbols.
        done = False
        found_precomp = set(precomputed.keys())
        while not done:
            done = True
            all_new = set()
            for sym in found_precomp:
                code_block = pre[sym]
                new = set([s for s in code_block.symbols
                            if s in pre and s not in precomputed])
                all_new.update(new)
            if len(all_new) > 0:
                done = False
                for s in all_new:
                    precomputed[s] = pre[s]
            found_precomp = all_new

        self.precomputed = sort_precomputed(precomputed)

        # Update the context.
        context = self.context
        for p, cb in self.precomputed.items():
            context[p] = cb.context[p]

    ##########################################################################
    # Public interface.
    ##########################################################################
    def update(self):
        self.context = Context()
        if not self.has_subgroups:
            self._setup_precomputed()

    def get_array_names(self, recompute=False):
        """Returns two sets of array names, the first being source_arrays
        and the second being destination array names.
        """
        if not recompute and self.src_arrays is not None:
            return set(self.src_arrays), set(self.dest_arrays)
        src_arrays = set()
        dest_arrays = set()
        for equation in self.equations:
            s, d = get_arrays_used_in_equation(equation)
            src_arrays.update(s)
            dest_arrays.update(d)

        for cb in self.precomputed.values():
            src_arrays.update(cb.src_arrays)
            dest_arrays.update(cb.dest_arrays)

        self.src_arrays = src_arrays
        self.dest_arrays = dest_arrays
        return src_arrays, dest_arrays

    def get_variable_names(self):
        # First get all the contexts and find the names.
        all_vars = set()
        for cb in self.precomputed.values():
            all_vars.update(cb.symbols)

        # Filter out all arrays.
        filtered_vars = [x for x in all_vars \
                         if not x.startswith(('s_', 'd_'))]
        # Filter other things.
        ignore = ['KERNEL', 'GRADIENT', 's_idx', 'd_idx']
        # Math functions.
        import math
        ignore += [x for x in dir(math) if not x.startswith('_')
                                            and callable(getattr(math, x))]
        try:
            ignore.remove('gamma')
            ignore.remove('lgamma')
        except ValueError:
            # Older Python's don't have gamma/lgamma.
            pass
        filtered_vars = [x for x in filtered_vars if x not in ignore]

        return filtered_vars

    def get_array_declarations(self, names, known_types={}):
        decl = []
        for arr in sorted(names):
            if arr in known_types:
                decl.append('cdef {type} {arr}'.format(
                    type=known_types[arr].type, arr=arr
                ))
            else:
                decl.append('cdef double* %s'%arr)
        return '\n'.join(decl)

    def get_variable_declarations(self, context):
        return self._get_variable_decl(context, mode='declare')

    def get_variable_array_setup(self):
        names = list(self.context.keys())
        names.sort()
        code = []
        for var in names:
            value = self.context[var]
            if isinstance(value, (list, tuple)):
                code.append(
                    '{var} = &_{var}.data[thread_id*aligned({size}, 8)]'\
                        .format(size=len(value), var=var)
                )
        return '\n'.join(code)

    def has_initialize(self):
        return self._has_code('initialize')

    def get_initialize_code(self, kernel=None):
        code = self._get_code(kind='initialize')
        return self._set_kernel(code, kernel)

    def has_loop(self):
        return self._has_code('loop')

    def get_loop_code(self, kernel=None):
        code = self._get_code(kind='loop')
        return self._set_kernel(code, kernel)

    def has_post_loop(self):
        return self._has_code('post_loop')

    def get_post_loop_code(self, kernel=None):
        code = self._get_code(kind='post_loop')
        return self._set_kernel(code, kernel)

    def has_reduce(self):
        return self._has_code('reduce')

    def get_reduce_code(self):
        return self._get_code(kind='reduce')

    def get_converged_condition(self):
        if self.has_subgroups:
            code = [g.get_converged_condition() for g in self.equations]
            return ' & '.join(code)
        else:
            code = []
            for equation in self.equations:
                code.append('(self.%s.converged() > 0)'%equation.var_name)
            # Note, we use '&' because we want to call converged on all equations.
            # and not be short-circuited by the first one that returns False.
            return ' & '.join(code)

    def get_equation_wrappers(self, known_types={}):
        classes = defaultdict(lambda: 0)
        eqs = {}
        for equation in self.equations:
            cls = equation.__class__.__name__
            n = classes[cls]
            equation.var_name = '%s%d'%(camel_to_underscore(equation.name), n)
            classes[cls] += 1
            eqs[cls] = equation
        wrappers = []
        predefined = dict(get_predefined_types(self.pre_comp))
        predefined.update(known_types)
        code_gen = CythonGenerator(known_types=predefined)
        for cls in sorted(classes.keys()):
            code_gen.parse(eqs[cls])
            wrappers.append(code_gen.get_code())
        return '\n'.join(wrappers)

    def get_equation_defs(self):
        lines = []
        for equation in self.equations:
            code = 'cdef public {cls} {name}'.format(cls=equation.name,
                                                     name=equation.var_name)
            lines.append(code)
        return '\n'.join(lines)

    def get_equation_init(self):
        lines = []
        for i, equation in enumerate(self.equations):
            code = 'self.{name} = {cls}(**equations[{idx}].__dict__)'\
                        .format(name=equation.var_name, cls=equation.name,
                                idx=i)
            lines.append(code)
        return '\n'.join(lines)
