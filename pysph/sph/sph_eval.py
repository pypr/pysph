from collections import defaultdict, OrderedDict
from mako.template import Template
from os.path import dirname, join

from pysph.sph.equations import Group
from pysph.base.ext_module import ExtModule

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
    props.difference_update(set(('tag', 'pid', 'gid')))
    array_names = ', '.join(sorted(props))
    return array_names    

###############################################################################
# `SPHEval` class.
###############################################################################
class SPHEval(object):
    def __init__(self, particle_arrays, equations, locator, kernel):
        self.particle_arrays = particle_arrays
        self.equation_groups = group_equations(equations)
        self.locator = locator
        self.kernel = kernel
        
        all_equations = []
        for group in self.equation_groups:
            all_equations.extend(group.equations)
        self.all_group = Group(equations=all_equations)
        
        self.groups = [self._make_group(g) for g in self.equation_groups]
        self._setup()
        
    ##########################################################################
    # Non-public interface.
    ##########################################################################
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
                if equation.dest != dest:
                    continue
                if equation.no_source:
                    eqs_with_no_source.append(equation)
                else:
                    for src in equation.sources:
                        sources[src].append(equation)
                        
            for src in sources:
                eqs = sources[src]
                sources[src] = Group(eqs)
                
            dests[dest] = (Group(eqs_with_no_source), sources)
            
        return dests
                
    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_helpers(self):
        helpers = []
        helpers.extend(get_code(self.kernel, 'helper'))
        
        helpers.extend(get_code(self.locator, 'helper'))
        
        return '\n'.join(helpers)
        
    def get_variable_declarations(self):
        group = self.all_group
        names = group.get_variable_names()
        ctx = group.check_variables(names)
        return group.get_variable_declarations(ctx)
        
    def get_array_declarations(self):
        group = self.all_group
        src, dest = group.get_array_names()
        src.update(dest)
        return group.get_array_declarations(src)
                
    def get_initialization(self, equation_group):
        group = equation_group
        names = group.get_variable_names()
        ctx = group.check_variables(names)
        return group.get_variable_initializations(ctx)
        
    def get_dest_array_setup(self, dest_name, eqs_with_no_source, sources):
        src, dest_arrays = eqs_with_no_source.get_array_names()
        for g in sources.values():
            s, d = g.get_array_names()
            dest_arrays.update(d)
        lines = ['NP_DEST = self.%s.size()'%dest_name]
        lines += ['%s = dst.%s.data'%(n, n[2:])
                 for n in dest_arrays]
        return '\n'.join(lines)
        
    def get_src_array_setup(self, src_name, eq_group):
        src_arrays, dest = eq_group.get_array_names()        
        lines = ['NP_SRC = self.%s.size()'%src_name]
        lines += ['%s = src.%s.data'%(n, n[2:])
                 for n in src_arrays]
        return '\n'.join(lines)
        
    def get_locator_code(self, src_name, dest_name):
        locator_name = self.locator.__class__.__name__
        return 'locator = %s(self.%s, self.%s)'%(locator_name, src_name, dest_name)
                
    def get_code(self):
        helpers = self.get_helpers()
        array_names =  get_array_names(self.particle_arrays)
        parrays = [pa.name for pa in self.particle_arrays]
        pa_names = ', '.join(parrays)
        locator = '\n'.join(get_code(self.locator, 'code'))
        path = join(dirname(__file__), 'sph_eval.mako')
        template = Template(filename=path)
        return template.render(helpers=helpers, array_names=array_names, 
                               pa_names=pa_names, locator=locator, object=self)

    def set_nnps(self, nnps):
        self.nnps = nnps
        self.calc.set_nnps(nnps)
        
    def compute(self):
        self.sph_compute()
