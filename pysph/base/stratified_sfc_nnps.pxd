from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

from nnps_base cimport *

ctypedef unsigned int u_int
ctypedef map[u_int, pair[u_int, u_int]] key_to_idx_t
ctypedef vector[u_int] u_int_vector_t

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef extern from 'math.h':
    double log(double) nogil
    double log2(double) nogil

cdef extern from "z_order.h":
    ctypedef unsigned long long uint64_t
    inline uint64_t get_key(uint64_t i, uint64_t j, uint64_t k) nogil

    cdef cppclass CompareSortWrapper:
        CompareSortWrapper() nogil except +
        CompareSortWrapper(double* x_ptr, double* y_ptr, double* z_ptr,
                double* xmin, double cell_size, u_int* current_pids,
                int length) nogil except +
        inline void compare_sort() nogil

cdef class StratifiedSFCNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef double radius_scale2

    cdef public int num_levels

    cdef double interval_size

    cdef u_int_vector_t*** pids
    cdef u_int_vector_t** current_pids

    cdef key_to_idx_t*** pid_indices
    cdef key_to_idx_t** current_indices

    cdef double** cell_sizes
    cdef double* current_cells

    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cpdef int count_particles(self, int interval)

    cpdef double get_binning_size(self, int interval)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper,
            int pa_index, UIntArray indices, u_int_vector_t** current_pids,
            key_to_idx_t** current_indices, double* current_cells)

    cdef inline int _get_level(self, double h) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


