# Automatically generated, do not edit.
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

cimport numpy
from pysph.base.carray cimport DoubleArray, LongArray, IntArray
from pysph.base.carray import DoubleArray, LongArray, IntArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.particle_array import ParticleArray

${helpers}

# #############################################################################
cdef class ParticleArrayWrapper:
    cdef public ParticleArray array
    cdef public IntArray tag, pid
    cdef public DoubleArray ${array_names}
    
    def __init__(self, pa):
        self.array = pa
        props = set(pa.properties.keys())
        props = props.union(['tag', 'pid'])
        for prop in props:
            setattr(self, prop, pa.get_carray(prop))
        
    cpdef long size(self):
        return self.array.get_number_of_particles()
        
        
${locator}

# #############################################################################
cdef class SPHCalc:
    cdef public ParticleArrayWrapper ${pa_names}
    
    def __init__(self, *particle_arrays):
        for pa in particle_arrays:
            name = pa.name
            setattr(self, name, ParticleArrayWrapper(pa))
    
    cpdef compute(self):
        cdef long s_idx, d_idx, NP_SRC, NP_DEST
        cdef LongArray nbrs = LongArray()
        #######################################################################
        ##  Declare all the arrays.
        #######################################################################
        # Arrays.\
        ${indent(object._get_array_declarations(), 2)}
        #######################################################################
        ## Declare any variables.
        #######################################################################
        # Variables.\
        ${indent(object._get_variable_declarations(), 2)}
        #######################################################################
        ## Iterate over groups:
        ## Groups are organized as {destination: (eqs_with_no_source, sources)}
        ## eqs_with_no_source: [equations] all SPH Equations with no source.
        ## sources are {source: [equations...]} 
        #######################################################################
        % for g_idx, group in enumerate(object.groups):
        # Group ${g_idx}.
        #######################################################################
        ## Iterate over destinations in this group.
        #######################################################################
        % for dest, (eqs_with_no_source, sources) in group.iteritems():
        # Destination ${dest}.\
        #######################################################################
        ## Setup destination array pointers.
        #######################################################################
        ${indent(object._get_dest_array_setup(dest, eqs_with_no_source, sources), 2)}
        #######################################################################
        ## Handle all the equations that do not have a source.
        #######################################################################
        % if len(eqs_with_no_source) > 0:
        # SPH Equations with no sources.
        for d_idx in range(NP_DEST):
            % for equation in eqs_with_no_source:
            # Equation ${equation.__class__.__name__}. \
            ${indent(object._get_equation_loop(equation), 3)}
            % endfor
        % endif
        #######################################################################
        ## Iterate over sources.
        #######################################################################
        % for source, equations in sources.iteritems():
        # Source ${source}.\
        #######################################################################
        ## Setup source array pointers.
        #######################################################################
        ${indent(object._get_src_array_setup(source, equations), 2)}
        # Locator.\
        #######################################################################
        ## Create the locator
        #######################################################################
        ${indent(object._get_locator_code(source, dest), 2)}
        #######################################################################
        ## Iterate over destination particles.
        #######################################################################
        for d_idx in range(NP_DEST):
            # Initialize temp vars.\
            ${indent(object._get_initialization(equations), 3)}
            ###################################################################
            ## Find and iterate over neighbors.
            ###################################################################
            locator.get_neighbors(d_idx, nbrs)
            for nbr_idx in range(nbrs.length):
                s_idx = nbrs[nbr_idx]
                ###############################################################
                ## Iterate over the equations for the same set of neighbors.
                ###############################################################
                % for equation in equations:
                # Equation ${equation.__class__.__name__}. \
                ${indent(object._get_equation_loop(equation), 4)}
                % endfor
            ###################################################################
            ## Do any post neighbor loop assignments.
            ###################################################################
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
        