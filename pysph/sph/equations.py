# System library imports.
import ast

from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from copy import deepcopy
import itertools
import numpy
from textwrap import dedent

# Local imports.
from ast_utils import get_symbols

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
    __getattr__ = dict.__getitem__
    def __setattr__(self, key, value):
        self[key] = value


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
        self.src_arrays = set(x for x in symbols
                                if x.startswith('s_') and x != 's_idx')
        self.dest_arrays = set(x for x in symbols
                                if x.startswith('d_') and x != 'd_idx')
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

    c.RHOIJ = BasicCodeBlock(code="RHOIJ = 0.5*(d_rho[d_idx] + s_rho[s_idx])", RHOIJ=0.0)

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
                code="WIJ = KERNEL(XIJ[0], XIJ[1], XIJ[2], HIJ)",
                WIJ=0.0)

    c.WI = BasicCodeBlock(
                code="WI = KERNEL(XIJ[0], XIJ[1], XIJ[2], d_h[d_idx])",
                WI=0.0)

    c.WJ = BasicCodeBlock(
                code="WJ = KERNEL(XIJ[0], XIJ[1], XIJ[2], s_h[s_idx])",
                WJ=0.0)

    c.DWIJ = BasicCodeBlock(
                code="GRADIENT(XIJ[0], XIJ[1], XIJ[2], HIJ, DWIJ)",
                DWIJ=[0.0, 0.0, 0.0])

    c.DWI = BasicCodeBlock(
                code="GRADIENT(XIJ[0], XIJ[1], XIJ[2], d_h[d_idx], DWI)",
                DWI=[0.0, 0.0, 0.0])

    c.DWJ = BasicCodeBlock(
                code="GRADIENT(XIJ[0], XIJ[1], XIJ[2], s_h[s_idx], DWJ)",
                DWJ=[0.0, 0.0, 0.0])
    return c


##############################################################################
# `CodeBlock` class.
##############################################################################
class CodeBlock(BasicCodeBlock):

    pre_comp = precomputed_symbols()

    ##########################################################################
    # Private interface.
    ##########################################################################
    def _setup_code(self, code):
        super(CodeBlock, self)._setup_code(code)

        # Calculate the precomputed symbols for this equation.
        pre = self.pre_comp
        precomputed = dict((s, pre[s]) for s in self.symbols if s in pre)

        # Now find the precomputed symbols in the pre-computed symbols.
        precomputed_symbols = set()
        for cb in precomputed.values():
            precomputed_symbols.update((s for s in cb.symbols if s in pre))

        for s in precomputed_symbols:
            if s not in precomputed:
                precomputed[s] = pre[s]

        self.precomputed = sort_precomputed(precomputed)

        # Update the context.
        context = self.context
        for p, cb in self.precomputed.iteritems():
            context[p] = cb.context[p]


def sort_precomputed(precomputed):
    """Sorts the precomputed equations in the given dictionary as per the
    dependencies of the symbols and returns an ordered dict.
    """
    weights = dict((x, None) for x in precomputed)
    pre_comp = CodeBlock.pre_comp
    # Find the dependent pre-computed symbols for each in the precomputed.
    depends = dict((x, None) for x in precomputed)
    for pre, cb in precomputed.iteritems():
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


##############################################################################
# `Equation` class.
##############################################################################
class Equation(object):

    ##########################################################################
    # `object` interface.
    ##########################################################################
    def __init__(self, dest, sources=None, name=None):
        self.dest = dest
        self.sources = sources if sources is not None and len(sources) > 0 \
                                                                else None
        # Does the equation require neighbors or not.
        self.no_source = self.sources is None
        self.name = self.__class__.__name__ if name is None else name

        # The initialization of any variables.  Should be a set of variables.
        self.initialize_vars = None
        # The loop code is run for each particle pair.
        self.loop = None
        # The post_loop code is run for each destination particle.
        self.post_loop = None
        self.setup()

    ##########################################################################
    # Public interface.
    ##########################################################################
    def setup(self):
        """setup the code and context.
        """
        pass

class VariableClashError(Exception):
    pass


###############################################################################
# `Group` class.
###############################################################################
class Group(object):
    """A group of equations.

    This class provides some support for the code generation for the
    collection of equations.
    """
    def __init__(self, equations):
        self.equations = equations
        self.src_arrays = self.dest_arrays = None
        self.update()

    ##########################################################################
    # Non-public interface.
    ##########################################################################
    def _get_variable_decl(self, context, mode='declare'):
        decl = []
        names = context.keys()
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
                    decl.append('cdef double[{size}] {var}'\
                            .format(size=len(value), var=var))
                else:
                    pass
        return '\n'.join(decl)

    def _get_code(self, kind='loop'):
        assert kind in ('loop', 'post_loop')
        # We assume here that precomputed quantities are only relevant
        # for loops and not post_loops.
        pre = []
        if kind == 'loop':
            for p, cb in self.precomputed.iteritems():
                pre.append(cb.code.strip())
            if len(pre) > 0:
                pre.append('')
        code = []
        for eq in self.equations:
            cb = getattr(eq, kind)
            if cb:
                code.append('# %s.'%eq.name)
                code.append(cb.code.strip())
        if len(code) > 0:
            code.append('')
        return '\n'.join(pre + code)

    def _set_kernel(self, code, kernel):
        if kernel is not None:
            k_name = kernel.__class__.__name__
            k_func = k_name + 'Kernel'
            g_func = k_name + 'Gradient'
            return code.replace('GRADIENT', g_func).replace('KERNEL', k_func)
        else:
            return code

    ##########################################################################
    # Public interface.
    ##########################################################################
    def update(self):
        precomp = {}
        initialize_vars = set()
        for equation in self.equations:
            if equation.initialize_vars:
                initialize_vars.update(equation.initialize_vars)
            if equation.loop:
                precomp.update(equation.loop.precomputed)
            if equation.post_loop:
                precomp.update(equation.post_loop.precomputed)

        self.precomputed = sort_precomputed(precomp)
        self.initialize_vars = initialize_vars

    def get_array_names(self, recompute=False):
        """Returns two sets of array names, the first being source_arrays
        and the second being destination array names.
        """
        if not recompute and self.src_arrays is not None:
            return set(self.src_arrays), set(self.dest_arrays)
        src_arrays = set()
        dest_arrays = set()
        for equation in self.equations:
            if equation.loop is not None:
                src_arrays.update(equation.loop.src_arrays)
                dest_arrays.update(equation.loop.dest_arrays)
            if equation.post_loop is not None:
                src_arrays.update(equation.post_loop.src_arrays)
                dest_arrays.update(equation.post_loop.dest_arrays)
        for cb in self.precomputed.values():
            src_arrays.update(cb.src_arrays)
            dest_arrays.update(cb.dest_arrays)

        self.src_arrays = src_arrays
        self.dest_arrays = dest_arrays
        return set(src_arrays), set(dest_arrays)

    def get_variable_names(self):
        # First get all the contexts and find the names.
        all_vars = set()
        for equation in self.equations:
            if equation.loop is not None:
                all_vars.update(equation.loop.symbols)
            if equation.post_loop is not None:
                all_vars.update(equation.post_loop.symbols)
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

    def check_variables(self, variables):
        """Given the sequence of variables, check for any conflicts
        in the equations and returns the full context if there are no clashes.
        """
        context = Context()
        seen = Context()
        for equation in self.equations:
            for var in variables:
                for eq_kind in (equation.loop, equation.post_loop):
                    if eq_kind is None or var not in eq_kind.context:
                        continue
                    eq_val = eq_kind.context[var]
                    if var in context:
                        ctx_val = context[var]
                        if type(eq_val) != type(ctx_val):
                            msg = 'Variable %s in %s has different type from '\
                                'equation %s'%(var, equation.name, seen[var])
                            raise VariableClashError(msg)
                        elif eq_val != ctx_val:
                            msg = 'Variable %s in %s has different value from '\
                                'equation %s'%(var, equation.name, seen[var])
                            raise VariableClashError(msg)
                    else:
                        context[var] = eq_val
                        seen[var] = equation.name
        return context

    def get_array_declarations(self, names):
        decl = []
        for arr in sorted(names):
            decl.append('cdef double* %s'%arr)
        return '\n'.join(decl)

    def get_variable_declarations(self, context):
        return self._get_variable_decl(context, mode='declare')

    def get_variable_initializations(self, context):
        # Only initialize variables that are to be initialized
        ctx = Context((x, context[x]) for x in context
                                        if x in self.initialize_vars)
        return self._get_variable_decl(ctx, mode='initialize')

    def get_loop_code(self, kernel=None):
        code = self._get_code()
        return self._set_kernel(code, kernel)

    def get_post_loop_code(self, kernel=None):
        code = self._get_code(kind='post_loop')
        return self._set_kernel(code, kernel)

##############################################################################
# `ContinuityEquation` class.
##############################################################################
class ContinuityEquation(Equation):
    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ=[0.0, 0.0, 0.0],
             VIJ=[0.0, 0.0, 0.0]):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class SummationDensity(Equation):
    def loop(self, d_idx, d_rho, s_idx, s_m, WIJ=0.0):
        d_rho[d_idx] += s_m[s_idx]*WIJ

class BodyForce(Equation):
    def __init__(self, dest, sources,
                 fx=0.0, fy=0.0, fz=0.0):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(BodyForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.fx
        d_av[d_idx] += self.fy
        d_aw[d_idx] += self.fz

class TaitEOS(Equation):
    def __init__(self, dest, sources,
                 rho0=1000.0, c0=1.0, gamma=7.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.B = rho0*c0*c0/gamma
        self.ratio = 0.0
        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        d_p[d_idx] = self.B * (tmp - 1.0)
        d_cs[d_idx] = self.c0 * pow( ratio, self.gamma1 )

class MomentumEquation(Equation):
    def __init__(self, dest, sources,
                 alpha=1.0, beta=1.0, eta=0.1, gx=0.0, gy=0.0, gz=0.0,
                 c0=1.0):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dt_fac = 0.0
        super(MomentumEquation, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ=[0.0, 0.0, 0.0],
             XIJ=[0.0, 0.0, 0.0], HIJ=1.0, R2IJ=1.0, RHOIJ1=1.0,
             DWIJ=[1.0, 1.0, 1.0]):
        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij)/(R2IJ + self.eta*self.eta*HIJ*HIJ)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        # compute the CFL time step factor
        _dt_fac = 0.0
        if R2IJ > 1e-12:
            _dt_fac = abs( HIJ * vijdotxij/R2IJ )
            self.dt_fac = max(_dt_fac, self.dt_fac)

        tmp = d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21

        d_au[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp + piij) * DWIJ[2]


    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] +=  self.gx
        d_av[d_idx] +=  self.gy
        d_aw[d_idx] +=  self.gz


class XSPHCorrection(Equation):
    def __init__(self, dest, sources, eps=0.5):
        self.eps = eps
        super(XSPHCorrection, self).__init__(dest, sources)

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, d_az, WIJ=1.0, RHOIJ1=1.0,
             VIJ=[1.0,1,1]):
        tmp = -self.eps * s_m[s_idx]*WIJ*RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]
        d_az[d_idx] += tmp * VIJ[2]


    def post_loop(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]
        d_az[d_idx] += d_w[d_idx]


if __name__ == '__main__':
    # Simple demonstration of one of the equations.
    e = ContinuityEquation(dest='fluid', sources=['fluid'])
    print e.loop(DWIJ=[1,1,1], XIJ=[1,1,1], s_m=[1,1])
