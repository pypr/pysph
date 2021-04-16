# Automatically generated, do not edit.
# cython: cdivision=True, language_level=3
# distutils: language=c++
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

<%def name="do_group(helper, group, level=0)" buffered="True">
#######################################################################
## Call any `pre` functions
#######################################################################
% if group.pre:
${indent(helper.get_pre_call(group), 0)}
% endif
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
${indent(helper.get_dest_array_setup(dest, eqs_with_no_source, sources, group), 0)}
dst_array_index = dst.index

#######################################################################
## Call py_initialize for all equations for this destination.
#######################################################################
${indent(all_eqs.get_py_initialize_code(), 0)}
#######################################################################
## Initialize all equations for this destination.
#######################################################################
% if all_eqs.has_initialize():
# Initialization for destination ${dest}.
for d_idx in ${helper.get_parallel_range(group)}:
    ${indent(all_eqs.get_initialize_code(helper.object.kernel), 1)}
% endif
#######################################################################
## Handle all the equations that do not have a source.
#######################################################################
% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
# SPH Equations with no sources.
for d_idx in ${helper.get_parallel_range(group)}:
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

% if eq_group.has_initialize_pair():
for d_idx in ${helper.get_parallel_range(group)}:
    ${indent(eq_group.get_initialize_pair_code(helper.object.kernel), 1)}
% endif

% if eq_group.has_loop() or eq_group.has_loop_all():
#######################################################################
## Iterate over destination particles.
#######################################################################
nnps.set_context(src_array_index, dst_array_index)

${helper.get_parallel_block()}
    thread_id = threadid()
    ${indent(eq_group.get_variable_array_setup(), 1)}
    for d_idx in ${helper.get_parallel_range(group, nogil=False)}:
        ###############################################################
        ## Find and iterate over neighbors.
        ###############################################################
        nnps.get_nearest_neighbors(d_idx, <UIntArray>self.nbrs[thread_id])
        NBRS = (<UIntArray>self.nbrs[thread_id]).data
        N_NBRS = (<UIntArray>self.nbrs[thread_id]).length
% if eq_group.has_loop_all():
        ${indent(eq_group.get_loop_all_code(helper.object.kernel), 2)}
% endif
% if eq_group.has_loop():
        for nbr_idx in range(N_NBRS):
            s_idx = <long>(NBRS[nbr_idx])
            ###########################################################
            ## Iterate over the equations for the same set of neighbors.
            ###########################################################
            ${indent(eq_group.get_loop_code(helper.object.kernel), 3)}
% endif ## if has_loop
% endif ## if eq_group.has_loop() or has_loop_all():
# Source ${source} done.
# --------------------------------------
% endfor
###################################################################
## Do any post_loop assignments for the destination.
###################################################################
% if all_eqs.has_post_loop():
# Post loop for destination ${dest}.
for d_idx in ${helper.get_parallel_range(group)}:
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
% endfor
#######################################################################
## Update NNPS locally if needed
#######################################################################
% if group.update_nnps:
# Updating NNPS.
with profile_ctx("Integrator.update_domain"):
    nnps.update_domain()
with profile_ctx("nnps.update"):
    nnps.update()
% endif
#######################################################################
## Call any `post` functions
#######################################################################
% if group.post:
${indent(helper.get_post_call(group), 0)}
% endif
</%def>

from libc.stdio cimport printf
from libc.math cimport *
from libc.math cimport fabs as abs
cimport numpy
import numpy
from cython import address
% if not helper.config.use_openmp:
from cython.parallel import threadid
prange = range
% else:
from cython.parallel import parallel, prange, threadid
% endif

from compyle.profile import profile_ctx
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps_base cimport NNPS
from pysph.base.reduce_array import serial_reduce_array
% if helper.object.mode == 'serial':
from pysph.base.reduce_array import dummy_reduce_array as parallel_reduce_array
% elif helper.object.mode == 'mpi':
from pysph.base.reduce_array import mpi_reduce_array as parallel_reduce_array
% endif

from pysph.base.nnps import get_number_of_threads
from cyarray.carray cimport (DoubleArray, FloatArray, IntArray, LongArray, UIntArray,
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
    cdef object groups
    cdef object all_equations
    ${indent(helper.get_kernel_defs(), 1)}
    ${indent(helper.get_equation_defs(), 1)}

    def __init__(self, kernel, equations, particle_arrays, groups):
        self.particle_arrays = tuple(particle_arrays)
        self.groups = groups
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
        all_equations = {}
        for equation in equations:
            all_equations[equation.var_name] = equation
        self.all_equations = all_equations

    def __dealloc__(self):
        aligned_free(self.nbrs)

    def set_nnps(self, NNPS nnps):
        self.nnps = nnps

    def update_particle_arrays(self, particle_arrays):
        for pa in particle_arrays:
            name = pa.name
            getattr(self, name).set_array(pa)

    cpdef compute(self, double t, double dt):
        cdef long nbr_idx, NP_SRC, NP_DEST, D_START_IDX
        cdef long s_idx, d_idx
        cdef int thread_id, N_NBRS
        cdef unsigned int* NBRS
        cdef NNPS nnps = self.nnps
        cdef ParticleArrayWrapper src, dst

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
        % if group.condition is not None:
        if ${helper.get_condition_call(group)}:
        <%
        indent_lvl = 3
        %>
        % else:
        <%
        indent_lvl = 2
        %>
        % endif
        % if group.iterate:
        ${indent(helper.get_iteration_init(group), indent_lvl)}
        <%
        indent_lvl += 1
        %>
        % endif
        ## ---------------------
        ## ---- Subgroups start
        % if group.has_subgroups:
        % for sg_idx, sub_group in enumerate(group.data):
        ${indent("# Doing subgroup " + str(sg_idx), indent_lvl)}
        % if sub_group.condition is not None:
        if ${helper.get_condition_call(sub_group)}:
        <%
        indent_lvl += 1
        %>
        % endif
        ${indent(do_group(helper, sub_group, indent_lvl), indent_lvl)}
        % if sub_group.condition is not None:
        <%
        indent_lvl -= 1
        %>
        % endif
        % endfor # (for sg_idx, sub_group in enumerate(group.data))
        ## ---- Subgroups done
        ## ---------------------
        % else:  # No subgroups
        ${indent(do_group(helper, group, indent_lvl), indent_lvl)}
        % endif  # (if group.has_subgroups)
        ## Check the iteration conditions
        % if group.iterate:
        ${indent(helper.get_iteration_check(group), indent_lvl)}
        % endif

        # Group ${g_idx} done.
        # ---------------------------------------------------------------------
        % endif # (if len(group.data) > 0)
        % endfor
