from collections import defaultdict, OrderedDict
from mako.template import Template

from equations import Group
from ext_module import ExtModule

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
        self._setup()
        
    def _setup(self):
        code = self.get_code()
        self.ext_mod = ExtModule(code, verbose=True)
        mod = self.ext_mod.load()
        self.calc = mod.SPHCalc(*self.particle_arrays)
        self.sph_compute = self.calc.compute
        
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
            eqs_with_no_source = [] # For equations that have no source.
            for equation in equations:
                if equation.no_source:
                    eqs_with_no_source.append(equation)
                else:
                    for src in equation.sources:
                        sources[src].append(equation)
            dests[dest] = (eqs_with_no_source, sources)
            
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
        consts = []
        var_names = defaultdict(list)

        tmp_names = defaultdict(list)
        tmp_declare = defaultdict(list)

        const_names = defaultdict(list)
        const_declare = defaultdict(list)
        
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

            c = code.get('constants', [])
            consts.extend(c)
            names = [(x.name, x.declare) for x in c]
            for name, declare in names:
                const_names[ name ].append( eq_name )
                const_declare[ name ].append( declare )

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
            vars = code.get('variables', [])
            tmps = code.get('temporaries', [])
            consts = code.get('constants', [])
            for var in vars:
                init[var.initialize] = None
            for var in tmps:
                init[var.initialize] = None
            for var in consts:
                init[var.initialize] = None            
            
        return '\n'.join(init.keys())
        
    def _get_equation_loop(self, equation):
        code = equation.cython_code().get('loop')
        kernel = self.kernel.__class__.__name__ + 'Kernel'
        gradient = self.kernel.__class__.__name__ + 'Gradient'
        code = code.replace('KERNEL', kernel).replace('GRADIENT', gradient)
        return code
        
    def _get_dest_array_setup(self, dest_name, eqs_with_no_source, sources):
        eqs = [eq for eq in eqs_with_no_source]
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
        
    def compute(self):
        self.sph_compute()
