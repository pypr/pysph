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
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
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
% for source, eq_group in sources.items():
# --------------------------------------
# Source ${source}.\
#######################################################################
## Setup source array pointers.
#######################################################################

src = self.${source}
${indent(helper.get_src_array_setup(source, eq_group), 0)}
src_array_index = src.index

% if eq_group.has_loop():
#######################################################################
## Iterate over destination particles.
#######################################################################
nnps.set_context(src_array_index, dst_array_index)

${helper.get_parallel_block()}
    thread_id = threadid()
    DT_ADAPT = &_DT_ADAPT.data[thread_id*aligned(3, 8)]
    ${indent(eq_group.get_variable_array_setup(), 1)}
    for d_idx in prange(NP_DEST):
        ###############################################################
        ## Find and iterate over neighbors.
        ###############################################################
        nbrs = NULL
        nnps.get_nearest_neighbors(d_idx, <UIntArray>self.nbrs[thread_id])
        for nbr_idx in range((<UIntArray>self.nbrs[thread_id]).length):
            s_idx = <int>((<UIntArray>self.nbrs[thread_id]).data[nbr_idx])
            ###########################################################
            ## Iterate over the equations for the same set of neighbors.
            ###########################################################
            ${indent(eq_group.get_loop_code(helper.object.kernel), 3)}

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
cimport numpy
import numpy
% if not helper.config.use_openmp:
from cython.parallel import threadid
prange = range
% else:
from cython.parallel import parallel, prange, threadid
% endif

from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps_base cimport NNPS
from pysph.base.reduce_array import serial_reduce_array
% if helper.object.mode == 'serial':
from pysph.base.reduce_array import dummy_reduce_array as parallel_reduce_array
% elif helper.object.mode == 'mpi':
from pysph.base.reduce_array import mpi_reduce_array as parallel_reduce_array
% endif

from pysph.base.nnps import get_number_of_threads
from pyzoltan.core.carray cimport (DoubleArray, FloatArray, IntArray, LongArray, UIntArray,
    aligned, aligned_free, aligned_malloc)

${helper.get_header()}

# #############################################################################
cdef class ParticleArrayWrapper:
    cdef public int index
    cdef public ParticleArray array
    ${indent(helper.get_array_decl_for_wrapper(), 1)}
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
    cdef public ParticleArrayWrapper ${helper.get_particle_array_names()}
    cdef public NNPS nnps
    cdef public int n_threads
    cdef public list _nbr_refs
    cdef void **nbrs
    # CFL time step conditions
    cdef public double dt_cfl, dt_force, dt_viscous
    ${indent(helper.get_kernel_defs(), 1)}
    ${indent(helper.get_equation_defs(), 1)}

    def __init__(self, kernel, equations, particle_arrays):
        self.particle_arrays = tuple(particle_arrays)
        self.n_threads = get_number_of_threads()
        cdef int i
        for i, pa in enumerate(particle_arrays):
            name = pa.name
            setattr(self, name, ParticleArrayWrapper(pa, i))

        self.nbrs = <void**>aligned_malloc(sizeof(void*)*self.n_threads)
        cdef UIntArray _arr
        self._nbr_refs = []
        for i in range(self.n_threads):
            _arr = UIntArray()
            _arr.reserve(1024)
            self.nbrs[i] = <void*>_arr
            self._nbr_refs.append(_arr)

        ${indent(helper.get_kernel_init(), 2)}
        ${indent(helper.get_equation_init(), 2)}

    def __dealloc__(self):
        aligned_free(self.nbrs)

    cdef _initialize_dt_adapt(self, double* DT_ADAPT):
        self.dt_cfl = self.dt_force = self.dt_viscous = -1e20
        cdef int i, _idx, offset
        cdef double* dta = DT_ADAPT
        offset = aligned(3, sizeof(double))
        for i in range(self.n_threads):
            _idx = i*offset
            dta[_idx + 0] = self.dt_cfl
            dta[_idx + 1] = self.dt_force
            dta[_idx + 2] = self.dt_viscous

    cdef _set_dt_adapt(self, double* DT_ADAPT):
        cdef int i, _idx, offset
        cdef double* dta = DT_ADAPT
        offset = aligned(3, sizeof(double))
        for i in range(self.n_threads):
            _idx = i*offset
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
        cdef int s_idx, d_idx, thread_id
        cdef NNPS nnps = self.nnps
        cdef ParticleArrayWrapper src, dst
        cdef DoubleArray _DT_ADAPT = DoubleArray(aligned(3, 8)*self.n_threads)
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
