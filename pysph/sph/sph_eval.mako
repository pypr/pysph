# Automatically generated, do not edit.
#cython: cdivision=True
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

from libc.math cimport pow, sqrt
cimport numpy
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPS

from pyzoltan.core.carray cimport DoubleArray, IntArray, UIntArray

${helpers}

# #############################################################################
cdef class ParticleArrayWrapper:
    cdef public int index
    cdef public ParticleArray array
    cdef public IntArray tag, pid
    cdef public UIntArray gid
    cdef public DoubleArray ${array_names}

    def __init__(self, pa, index):
        self.index = index
        self.array = pa
        props = set(pa.properties.keys())
        props = props.union(['tag', 'pid', 'gid'])
        for prop in props:
            setattr(self, prop, pa.get_carray(prop))

    cpdef long size(self):
        return self.array.get_number_of_particles()


# #############################################################################
cdef class SPHCalc:
    cdef public ParticleArrayWrapper ${pa_names}
    cdef public NNPS nnps
    cdef UIntArray nbrs
    # CFL time step conditions
    cdef public double dt_cfl
    ${indent(object.get_kernel_defs(), 1)}
    ${indent(object.get_equation_defs(), 1)}

    def __init__(self, kernel, equations, *particle_arrays):
        for i, pa in enumerate(particle_arrays):
            name = pa.name
            setattr(self, name, ParticleArrayWrapper(pa, i))

        self.nbrs = UIntArray()
        ${indent(object.get_kernel_init(), 2)}
        ${indent(object.get_equation_init(), 2)}

    def set_nnps(self, NNPS nnps):
        self.nnps = nnps

    cpdef compute(self, double t, double dt):
        cdef long nbr_idx, NP_SRC, NP_DEST
        cdef int s_idx, d_idx
        cdef UIntArray nbrs = self.nbrs
        cdef NNPS nnps = self.nnps
        cdef ParticleArrayWrapper src, dst

        #######################################################################
        ##  Declare all the arrays.
        #######################################################################
        # Arrays.\
        ${indent(object.get_array_declarations(), 2)}
        #######################################################################
        ## Declare any variables.
        #######################################################################
        # Variables.\

        cdef int src_array_index, dst_array_index
        ${indent(object.get_variable_declarations(), 2)}
        #######################################################################
        ## Iterate over groups:
        ## Groups are organized as {destination: (eqs_with_no_source, sources, all_eqs)}
        ## eqs_with_no_source: Group([equations]) all SPH Equations with no source.
        ## sources are {source: Group([equations...])}
        ## all_eqs is a Group of all equations having this destination.
        #######################################################################
        % for g_idx, group in enumerate(object.groups):
        # ---------------------------------------------------------------------
        # Group ${g_idx}.
        #######################################################################
        ## Iterate over destinations in this group.
        #######################################################################
        % for dest, (eqs_with_no_source, sources, all_eqs) in group.iteritems():
        # ---------------------------------------------------------------------
        # Destination ${dest}.\
        #######################################################################
        ## Setup destination array pointers.
        #######################################################################

        dst = self.${dest}
        ${indent(object.get_dest_array_setup(dest, eqs_with_no_source, sources), 2)}
        dst_array_index = dst.index

        #######################################################################
        ## Initialize all equations for this destination.
        #######################################################################
        % if all_eqs.has_initialize():
        # Initialization for destination ${dest}.
        for d_idx in range(NP_DEST):
            ${indent(all_eqs.get_initialize_code(object.kernel), 3)}
        % endif
        #######################################################################
        ## Handle all the equations that do not have a source.
        #######################################################################
        % if len(eqs_with_no_source.equations) > 0:
        # SPH Equations with no sources.
        for d_idx in range(NP_DEST):
            ${indent(eqs_with_no_source.get_loop_code(object.kernel), 3)}
        % endif
        #######################################################################
        ## Iterate over sources.
        #######################################################################
        % for source, eq_group in sources.iteritems():
        # --------------------------------------
        # Source ${source}.\
        #######################################################################
        ## Setup source array pointers.
        #######################################################################

        src = self.${source}
        ${indent(object.get_src_array_setup(source, eq_group), 2)}
        src_array_index = src.index

        #######################################################################
        ## Iterate over destination particles.
        #######################################################################
        for d_idx in range(NP_DEST):
            ###################################################################
            ## Find and iterate over neighbors.
            ###################################################################
            nnps.get_nearest_particles(
                src_array_index, dst_array_index, d_idx, nbrs)

            for nbr_idx in range(nbrs._length):
                s_idx = <int>nbrs.data[nbr_idx]
                ###############################################################
                ## Iterate over the equations for the same set of neighbors.
                ###############################################################
                ${indent(eq_group.get_loop_code(object.kernel), 4)}

        # Source ${source} done.
        # --------------------------------------
        % endfor
        ###################################################################
        ## Do any post_loop assignments for the destination.
        ###################################################################
        % if all_eqs.has_post_loop():
        # Post loop for destination ${dest}.
        for d_idx in range(NP_DEST):
            ${indent(all_eqs.get_post_loop_code(object.kernel), 3)}
        % endif
        # Destination ${dest} done.
        # ---------------------------------------------------------------------
        % endfor
        # Group ${g_idx} done.
        # ---------------------------------------------------------------------
        % endfor
