# Automatically generated, do not edit.
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

from pysph.base.carray cimport DoubleArray, LongArray, IntArray, UIntArray
from pysph.base.carray import DoubleArray, LongArray, IntArray, UIntArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.particle_array import ParticleArray

${helpers}

# #############################################################################
cdef class ParticleArrayWrapper:
    cdef public ParticleArray array
    cdef public LongArray tag, group
    cdef public IntArray local, pid
    cdef public DoubleArray ${array_names}
    
    def __init__(self, pa):
        self.array = pa
        props = set(pa.properties.keys())
        props = props.union(['tag', 'group', 'local', 'pid'])
        for prop in props:
            setattr(self, prop, pa.get_carray(prop))
        
    cpdef long size(self):
        return self.array.get_number_of_particles()
        

# #############################################################################
cdef class SPHCalc:
    cdef public ParticleArrayWrapper ${pa_names}
    
    def __init__(self, *particle_arrays):
        for pa in particle_arrays:
            name = pa.name
            setattr(self, name, ParticleArrayWrapper(pa))
    
    cpdef void compute(self):
        cdef public long s_idx, d_idx, NP_SRC, NP_DEST
        cdef public LongArray nbrs
        # Arrays.\
        ${indent(object._get_array_declarations(), 2)}
        # Variables.\
        ${indent(object._get_variable_declarations(), 2)}
        % for g_idx, group in enumerate(object.groups):
        # Group ${g_idx}.
        % for dest, sources in group.iteritems():
        # Destination ${dest}.\
        ${indent(object._get_dest_array_setup(dest, sources), 2)}
        % for source, equations in sources.iteritems():
        # Source ${source}.\
        ${indent(object._get_src_array_setup(source, equations), 2)}
        # Locator.\
        ${indent(object.locator.cython_code().get('setup'), 2)}
        for d_idx in range(NP_DEST):
            # Initialize temp vars.\
            ${indent(object._get_initialization(equations), 3)}
            locator.get_neighbors(d_idx, nbrs)
            for nbr_idx in range(nbrs.length):
                s_idx = nbrs[nbr_idx]
                % for equation in equations:
                # Equation ${equation.__class__.__name__} \
                ${indent(object._get_equation_loop(equation), 4)}
                % endfor
            # Set destination values.\
            % for equation in equations:
            ${indent(equation.cython_code().get('post', ''), 3)}
            % endfor
        # Source ${source} done.
        % endfor 
        # Destination ${dest} done.
        % endfor
        # Group ${g_idx} done.
        % endfor
        