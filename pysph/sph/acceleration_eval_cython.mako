# Automatically generated, do not edit.
#cython: cdivision=True
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

<%def name="do_group(helper, group, level=0)" buffered="True">
#######################################################################
## Iterate over destinations in this group.
#######################################################################
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.iteritems():
# ---------------------------------------------------------------------
# Destination ${dest}.\
#######################################################################
## Setup destination array pointers.
#######################################################################

dst = self.${dest}
${indent(helper.get_dest_array_setup(dest, eqs_with_no_source, sources, group.real), 0)}
dst_array_index = dst.index

#######################################################################
## Initialize all equations for this destination.
#######################################################################
% if all_eqs.has_initialize():
# Initialization for destination ${dest}.
for d_idx in range(NP_DEST):
    ${indent(all_eqs.get_initialize_code(helper.object.kernel), 1)}
% endif
#######################################################################
## Handle all the equations that do not have a source.
#######################################################################
% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
# SPH Equations with no sources.
for d_idx in range(NP_DEST):
    ${indent(eqs_with_no_source.get_loop_code(helper.object.kernel), 1)}
% endif
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
${indent(helper.get_src_array_setup(source, eq_group), 0)}
src_array_index = src.index

% if eq_group.has_loop():
% if helper.object.cell_iteration:
#######################################################################
## Iterate over cells.
#######################################################################
ncells = nnps.get_number_of_cells()
for cell_index in range(ncells):
    # Find potential neighbors from nearby cells for this cell.
    nnps.get_particles_in_neighboring_cells(
        cell_index, src_array_index, potential_nbrs
    )

    # Get the indices for each particle in this cell
    nnps.get_particles_in_cell(
        cell_index, dst_array_index, particle_indices
    )
    for _index in range(particle_indices.length):
        # The destination index.
        d_idx = particle_indices[_index]
        # Get the neigbors for this destination index.
        nnps.get_nearest_particles_filtered(
            src_array_index, dst_array_index, d_idx,
            potential_nbrs, nbrs
        )

        # Now iterate over the neighbors.
        for nbr_idx in range(nbrs.length):
            s_idx = <int>nbrs.data[nbr_idx]
            ###########################################################
            ## Iterate over equations for the same set of neighbors.
            ###########################################################
            ${indent(eq_group.get_loop_code(helper.object.kernel), 3)}
% else:
#######################################################################
## Iterate over destination particles.
#######################################################################
nnps.set_context(src_array_index, dst_array_index)

with nogil, parallel():
    DT_ADAPT = &_DT_ADAPT.data[threadid()*3]
    ${indent(eq_group.get_variable_array_setup(), 1)}
    for d_idx in prange(NP_DEST):
        n_neighbors = 0
        ###############################################################
        ## Find and iterate over neighbors.
        ###############################################################
        nbrs = nnps.get_nearest_neighbors_raw(d_idx, &n_neighbors)
        for nbr_idx in range(n_neighbors):
            s_idx = <int>nbrs[nbr_idx]
            ###########################################################
            ## Iterate over the equations for the same set of neighbors.
            ###########################################################
            ${indent(eq_group.get_loop_code(helper.object.kernel), 3)}

% endif
% endif ## if eq_group.has_loop():
# Source ${source} done.
# --------------------------------------
% endfor
###################################################################
## Do any post_loop assignments for the destination.
###################################################################
% if all_eqs.has_post_loop():
# Post loop for destination ${dest}.
DT_ADAPT = _DT_ADAPT.data
for d_idx in range(NP_DEST):
    ${indent(all_eqs.get_post_loop_code(helper.object.kernel), 1)}
% endif

###################################################################
## Do any reductions for the destination.
###################################################################
% if all_eqs.has_reduce():
${indent(all_eqs.get_reduce_code(), 0)}
% endif

# Destination ${dest} done.
# ---------------------------------------------------------------------

#######################################################################
## Update NNPS locally if needed
#######################################################################
% if group.update_nnps:
# Updating NNPS.
nnps.update_domain()
nnps.update()
% endif

% endfor
</%def>

from libc.math cimport *
from libc.math cimport fabs as abs
from libc.math cimport M_PI as pi
from cython.parallel import parallel, prange, threadid
cimport numpy
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPS
from pysph.base.reduce_array import serial_reduce_array
% if helper.object.mode == 'serial':
from pysph.base.reduce_array import dummy_reduce_array as parallel_reduce_array
% elif helper.object.mode == 'mpi':
from pysph.base.reduce_array import mpi_reduce_array as parallel_reduce_array
% endif

from pysph.base.nnps import get_number_of_threads
from pyzoltan.core.carray cimport DoubleArray, IntArray, UIntArray

${header}

# #############################################################################
cdef class ParticleArrayWrapper:
    cdef public int index
    cdef public ParticleArray array
    cdef public IntArray tag, pid
    cdef public UIntArray gid
    cdef public DoubleArray ${array_names}
    cdef public str name

    def __init__(self, pa, index):
        self.index = index
        self.set_array(pa)

    cpdef set_array(self, pa):
        self.array = pa
        props = set(pa.properties.keys())
        props = props.union(['tag', 'pid', 'gid'])
        for prop in props:
            setattr(self, prop, pa.get_carray(prop))
        for prop in pa.constants.keys():
            setattr(self, prop, pa.get_carray(prop))

        self.name = pa.name

    cpdef long size(self, bint real=False):
        return self.array.get_number_of_particles(real)


# #############################################################################
cdef class AccelerationEval:
    cdef public tuple particle_arrays
    cdef public ParticleArrayWrapper ${pa_names}
    cdef public NNPS nnps
    cdef public int n_threads
    cdef UIntArray nbrs
    # CFL time step conditions
    cdef public double dt_cfl, dt_force, dt_viscous
    ${indent(helper.get_kernel_defs(), 1)}
    ${indent(helper.get_equation_defs(), 1)}

    def __init__(self, kernel, equations, particle_arrays):
        self.particle_arrays = tuple(particle_arrays)
        self.n_threads = get_number_of_threads()
        for i, pa in enumerate(particle_arrays):
            name = pa.name
            setattr(self, name, ParticleArrayWrapper(pa, i))

        self.nbrs = UIntArray()
        ${indent(helper.get_kernel_init(), 2)}
        ${indent(helper.get_equation_init(), 2)}

    cdef _initialize_dt_adapt(self, double* DT_ADAPT):
        self.dt_cfl = self.dt_force = self.dt_viscous = -1e20
        cdef int i, _idx
        cdef double* dta = DT_ADAPT
        for i in range(self.n_threads):
            _idx = i*3
            dta[_idx + 0] = self.dt_cfl
            dta[_idx + 1] = self.dt_force
            dta[_idx + 2] = self.dt_viscous

    cdef _set_dt_adapt(self, double* DT_ADAPT):
        cdef int i, _idx
        cdef double* dta = DT_ADAPT
        for i in range(self.n_threads):
            _idx = i*3
            dta[0] = max(dta[0], dta[_idx + 0])
            dta[1] = max(dta[1], dta[_idx + 1])
            dta[2] = max(dta[2], dta[_idx + 2])

        self.dt_cfl = dta[0]
        self.dt_force = dta[1]
        self.dt_viscous = dta[2]

    def set_nnps(self, NNPS nnps):
        self.nnps = nnps

    def update_particle_arrays(self, particle_arrays):
        for pa in particle_arrays:
            name = pa.name
            getattr(self, name).set_array(pa)

    cpdef compute(self, double t, double dt):
        cdef long nbr_idx, NP_SRC, NP_DEST
        cdef int s_idx, d_idx
        cdef unsigned int* nbrs
        cdef UIntArray particle_indices = UIntArray(1000)
        cdef UIntArray potential_nbrs = UIntArray(1000)
        cdef int cell_index, ncells, _index
        cdef NNPS nnps = self.nnps
        cdef ParticleArrayWrapper src, dst
        cdef size_t n_neighbors
        cdef DoubleArray _DT_ADAPT = DoubleArray(3*self.n_threads)
        self._initialize_dt_adapt(_DT_ADAPT.data)
        cdef double* DT_ADAPT = _DT_ADAPT.data

        cdef int max_iterations, min_iterations, _iteration_count

        #######################################################################
        ##  Declare all the arrays.
        #######################################################################
        # Arrays.\
        ${indent(helper.get_array_declarations(), 2)}
        #######################################################################
        ## Declare any variables.
        #######################################################################
        # Variables.\

        cdef int src_array_index, dst_array_index
        ${indent(helper.get_variable_declarations(), 2)}
        #######################################################################
        ## Iterate over groups:
        ## Groups are organized as {destination: (eqs_with_no_source, sources, all_eqs)}
        ## eqs_with_no_source: Group([equations]) all SPH Equations with no source.
        ## sources are {source: Group([equations...])}
        ## all_eqs is a Group of all equations having this destination.
        #######################################################################
        % for g_idx, group in enumerate(helper.object.mega_groups):
        % if len(group.data) > 0: # No equations in this group.
        # ---------------------------------------------------------------------
        # Group ${g_idx}.
        % if group.iterate:
        max_iterations = ${group.max_iterations}
        min_iterations = ${group.min_iterations}
        _iteration_count = 1
        while True:
        % else:
        if True:
        % endif

            % if group.has_subgroups:
            % for sg_idx, sub_group in enumerate(group.data):
            # Doing subgroup ${sg_idx}
            ${indent(do_group(helper, sub_group, 3), 3)}
            % endfor

            % else:
            ${indent(do_group(helper, group, 3), 3)}
            % endif
            #######################################################################
            ## Break the iteration for the group.
            #######################################################################
            % if group.iterate:
            # Check for convergence or timeout
            if (_iteration_count >= min_iterations) and (${group.get_converged_condition()} or (_iteration_count == max_iterations)):
                _iteration_count = 1
                break
            _iteration_count += 1
            % endif

        # Group ${g_idx} done.
        # ---------------------------------------------------------------------
        % endif # (if len(group.data) > 0)
        % endfor
        self._set_dt_adapt(_DT_ADAPT.data)
