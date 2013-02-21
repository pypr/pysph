from collections import defaultdict, OrderedDict
from mako.template import Template
from textwrap import dedent
from os.path import basename, dirname, splitext

###############################################################################
# `CubicSpline` class.
###############################################################################
class CubicSpline(object):
    def cython_code(self):
        code = dedent('''\
        cdef double CubicSplineKernel(double x, double y, double h):
            return 1.0
        ''')
        return dict(helper=code)


###############################################################################
# `Equation` class.
###############################################################################
class Equation(object):
    def __init__(self, dest, sources):
        self.dest = dest
        self.sources = sources
        
###############################################################################
# `Group` class.
###############################################################################
class Group(object):
    def __init__(self, equations):
        self.equations = equations
        
###############################################################################
# `Variable` class.
###############################################################################
class Variable(object):
    """Should be unique to each equation.
    """
    def __init__(self, type, name, default=None):
        self.type = type
        self.name = name
        self.default = default
        if default is not None:
            declare = 'cdef {0} {1} = {2}'.format(self.type, self.name, 
                                                self.default)
            initialize = '{0} = {1}'.format(self.name, self.default)
        else:
            declare = 'cdef {0} {1}'.format(self.type, self.name)
            initialize = ''
        self.declare = declare
        self.initialize = initialize
        
###############################################################################
# `Temporary` class.
###############################################################################
class Temporary(Variable):
    """Use for temporary variables.  Can be common between equations.
    """
    pass
    

###############################################################################
# `SummationDensity` class.
###############################################################################
class SummationDensity(Equation):
    def cython_code(self):
        variables = [Variable(type='double', name='rho_sum', default=0.0)]
        temp = [Temporary(type='double', name='hab', default=0.0)]
        arrays = ['s_h', 's_m', 's_x', 'd_h', 'd_x', 'd_rho']
        
        loop = dedent('''\
        hab = 0.5*(s_h[s_idx] + d_h[d_idx])
        rho_sum += s_m[s_idx]*KERNEL(d_x[d_idx], s_x[s_idx], hab)
        ''')
        post = dedent('''\
        d_rho[d_idx] = rho_sum
        ''')
        return dict(variables=variables, temporaries=temp, loop=loop, post=post,
                    arrays=arrays)
 
###############################################################################
# `Locator` class.
###############################################################################
class Locator(object):
    def initialize(self):
        raise NotImplementedError()
        
    def get_neighbors(self, d_idx, nbr_array):
        raise NotImplementedError()
        
    def cython_code(self):
        raise NotImplementedError
        
        
###############################################################################
# `AllPairLocator` class.
###############################################################################
class AllPairLocator(Locator):
    def initialize(self):
        self.len = len(self.s_x)
        
    def cython_code(self):
        code = dedent('''\
        cdef class AllPairLocator:
            cdef ParticleArrayWrapper src, dest
            cdef long N
            cdef LongArray nbrs
            def __init__(self, ParticleArrayWrapper src, 
                         ParticleArrayWrapper dest):
                self.src = src
                self.dest = dest
                self.N = src.size()
                self.nbrs = LongArray(self.N)
                cdef long i
                for i in range(self.N):
                    self.nbrs[i] = i
                
            def get_neighbors(self, long d_idx, LongArray nbr_array):
                nbr_array.resize(self.N)
                nbr_array.set_data(self.nbrs.get_npy_array())
        '''
        )
        return dict(helper='', code=code)
        
###############################################################################
# `VariableNameClashError` class.
###############################################################################
class VariableNameClashError(Exception):
    pass

###############################################################################
def check_equations(equations):
    only_groups = [x for x in equations if isinstance(x, Group)]
    if len(only_groups) > 0 and len(only_groups) != len(equations):
        raise ValueError('All elements must be Groups if you use groups.')
    if len(only_groups) == 0:
        return [Group(equations)]
    else:
        return equations
        
###############################################################################
def get_code(obj, key):
    code = obj.cython_code()
    doc = '# From %s'%obj.__class__.__name__
    src = code.get(key, '')
    return [doc, src] if len(src) > 0 else []
    
###############################################################################
def get_array_names(particle_arrays):
    """Get the union of the names of all particle array properties.
    """
    props = set()
    for array in particle_arrays:
        for name in array.properties.keys():
            props.add(name)
    props.difference_update(set(('tag', 'group', 'local', 'pid', 'idx')))
    array_names = ', '.join(sorted(props))
    return array_names    

###############################################################################
# `SPHEval` class.
###############################################################################
class SPHEval(object):
    def __init__(self, particle_arrays, equations, locator, kernel):
        self.particle_arrays = particle_arrays
        self.equation_groups = check_equations(equations)
        self.locator = locator
        self.kernel = kernel
        self.groups = [self._make_group(g) for g in self.equation_groups]
        self.sph_compute = None
        
    def _make_group(self, group):
        equations = group.equations
        dest_list = []
        for equation in equations:
            dest = equation.dest
            if dest not in dest_list:
                dest_list.append(dest)
        
        dests = OrderedDict()
        for dest in dest_list:
            sources = defaultdict(list)
            for equation in equations:
                for src in equation.sources:
                    sources[src].append(equation)
            dests[dest] = sources
            
        return dests
                
    def _get_helpers(self):
        helpers = []
        helpers.extend(get_code(self.kernel, 'helper'))
        
        for group in self.equation_groups:
            for eq in group.equations:
                helpers.extend(get_code(eq, 'helper'))
                
        helpers.extend(get_code(self.locator, 'helper'))
        
        return '\n'.join(helpers)
        
    def _check_and_get_variables(self):
        vars = []
        temps = []
        var_names = defaultdict(list)
        tmp_names = defaultdict(list)
        tmp_declare = defaultdict(list)
        equations = []
        for g in self.equation_groups:
            equations.extend(g.equations)
            
        for equation in equations:
            eq_name = equation.__class__.__name__
            code = equation.cython_code()
            v = code.get('variables', [])
            vars.extend(v)
            names = [x.name for x in v]
            for name in names:
                var_names[name].append(eq_name)
            
            t = code.get('temporaries', [])
            temps.extend(t)
            names = [(x.name, x.declare) for x in t]
            for name, declare in names:
                tmp_names[name].append(eq_name)
                tmp_declare[name].append(declare)

        for name, eqs in var_names.iteritems():
            if len(eqs) > 1:
                msg = 'Variable %s defined in %s.'%(name, eqs)
                raise VariableNameClashError(msg)
                
        for name, eqs in tmp_names.iteritems():
            if name in var_names:
                msg = 'Temporary %s in equation %s also defined as variable '\
                      'in %s'%(name, eqs, var_names[name])
                raise VariableNameClashError(msg)
                
        for name, eqs in tmp_names.iteritems():
            if len(eqs) > 1:
                declares = tmp_declare[name]
                if not all(map(lambda v: v == declares[0], declares)):
                    msg = "Temporary declarations for %s in %s differ"%\
                            (name, eqs)
                    raise VariableNameClashError(msg)
        return vars, temps
        
    def _get_variable_declarations(self):
        vars, tmps = self._check_and_get_variables()
        decl = {}
        for var in vars:
            decl[var.declare] = None
        for var in tmps:
            decl[var.declare] = None
        return '\n'.join(decl.keys())
        
    def _get_array_declarations(self):
        equations = []
        for g in self.equation_groups:
            equations.extend(g.equations)
            
        decl = {}
        for eq in equations:
            code = eq.cython_code()
            for array in code.get('arrays'):
                src = 'cdef double* %s'%array
                decl[src] = None
        return '\n'.join(decl.keys())
                
    def _get_initialization(self, equations):
        init = {}
        for equation in equations:
            code = equation.cython_code()
            vars = code.get('variables')
            tmps = code.get('temporaries')
            for var in vars:
                init[var.initialize] = None
            for var in tmps:
                init[var.initialize] = None
            
        return '\n'.join(init.keys())
        
    def _get_equation_loop(self, equation):
        code = equation.cython_code().get('loop')
        kernel = self.kernel.__class__.__name__ + 'Kernel'
        gradient = self.kernel.__class__.__name__ + 'Gradient'
        code = code.replace('KERNEL', kernel).replace('GRADIENT', gradient)
        return code
        
    def _get_dest_array_setup(self, dest_name, sources):
        eqs = []
        for eq in sources.values():
            eqs.extend(eq)
        eqs = set(eqs)
        names = [arr for e in eqs for arr in e.cython_code().get('arrays') 
                    if arr.startswith('d_')]
        names = set(names)
        lines = ['NP_DEST = self.%s.size()'%dest_name]
        lines += ['%s = self.%s.%s.get_data_ptr()'%(n, dest_name, n[2:]) 
                 for n in names]
        return '\n'.join(lines)
        
    def _get_src_array_setup(self, src_name, eqs):
        names = [arr for e in eqs for arr in e.cython_code().get('arrays') 
                    if arr.startswith('s_')]
        names = set(names)
        lines = ['NP_SRC = self.%s.size()'%src_name]
        lines += ['%s = self.%s.%s.get_data_ptr()'%(n, src_name, n[2:]) 
                 for n in names]
        return '\n'.join(lines)
        
    def _get_locator_code(self, src_name, dest_name):
        locator_name = self.locator.__class__.__name__
        return 'locator = %s(self.%s, self.%s)'%(locator_name, src_name, dest_name)
                
    def get_code(self):
        helpers = self._get_helpers()
        array_names =  get_array_names(self.particle_arrays)
        parrays = [pa.name for pa in self.particle_arrays]
        pa_names = ', '.join(parrays)
        locator = '\n'.join(get_code(self.locator, 'code'))
        template = Template(filename='sph_eval.mako')
        return template.render(helpers=helpers, array_names=array_names, 
                               pa_names=pa_names, locator=locator, object=self)
        
    def generate(self, fp):
        code = self.get_code()
        fp.write(code)
        
    def build(self, fname):
        from pyximport import pyxbuild
        from distutils.extension import Extension
        import pysph
        import numpy
        inc_dirs = [dirname(dirname(pysph.__file__)), numpy.get_include(), '.']
        ext = Extension(name=splitext(fname)[0], sources=[fname], 
                        include_dirs=inc_dirs)
        return pyxbuild.pyx_to_dll(fname, ext)
        
    def load_mod(self, name, mod_path):
        import imp
        file, path, desc = imp.find_module(name, [dirname(mod_path)])
        mod = imp.load_module(name, file, path, desc)
        calc = mod.SPHCalc(*self.particle_arrays)
        self.sph_compute = calc.compute
        
    def setup_calc(self, fname):
        self.generate(open(fname, 'w'))
        name = splitext(basename(fname))[0]
        m_path = self.build(fname)
        self.load_mod(name, m_path)
        
    def compute(self):
        self.sph_compute()
